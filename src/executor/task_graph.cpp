/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gqe/executor/task_graph.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/device_properties.hpp>
#include <gqe/executor/aggregate.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/fetch.hpp>
#include <gqe/executor/filter.hpp>
#include <gqe/executor/gen_ident_col.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/partition.hpp>
#include <gqe/executor/partition_merge.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/executor/sort.hpp>
#include <gqe/executor/spill.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/executor/window.hpp>
#include <gqe/executor/write.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/gen_ident_col.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/shuffle.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/linux.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_device.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <set>
#include <stack>
#include <stdexcept>
#include <thread>

namespace gqe {

task_graph_builder::task_graph_builder(context_reference ctx_ref,
                                       storage::table_provider const* provider)
  : _ctx_ref(ctx_ref), _table_provider(provider)
{
}

std::unique_ptr<task_graph> task_graph_builder::build(physical::relation* root_relation)
{
  auto generated_tasks = generate_tasks(root_relation);

  // Update _stage_root_tasks with tasks in the last stage
  insert_pipeline_breaker(generated_tasks);

  return std::make_unique<task_graph>(task_graph{
    std::move(generated_tasks), std::move(_stage_root_tasks), std::move(_write_statistics)});
}

std::vector<std::shared_ptr<task>> task_graph_builder::generate_tasks(physical::relation* relation)
{
  generate_task_graph_visitor visitor(this);
  relation->accept(visitor);
  return std::move(visitor._generated_tasks);
}

std::shared_ptr<task> task_graph_builder::concatenate(
  std::vector<std::shared_ptr<task>> input_tasks, bool is_pipeline_breaker)
{
  if (input_tasks.size() == 1) {
    return std::move(input_tasks[0]);
  } else {
    if (is_pipeline_breaker) insert_pipeline_breaker(input_tasks);

    auto concatenated_input = std::make_shared<concatenate_task>(
      _ctx_ref, _current_task_id, _current_stage_id, std::move(input_tasks));
    _current_task_id++;
    return concatenated_input;
  }
}

void execute_task_graph_single_gpu(context_reference ctx_ref,
                                   task_graph const* task_graph_to_execute)
{
  auto task_manager_context =
    dynamic_cast<multi_process_task_manager_context*>(ctx_ref._task_manager_context);
  if (task_manager_context != nullptr && task_manager_context->comm->rank() != 0) {
    GQE_LOG_INFO("Process with rank {} is not the root process, skipping task graph execution.",
                 task_manager_context->comm->rank());
    return;
  }

  assert(ctx_ref._task_manager_context != nullptr);
  assert(ctx_ref._query_context != nullptr);
  assert(task_graph_to_execute != nullptr);

  // Cache the process-wide CPU affinity before any thread pinning.
  utility::set_thread_affinity_fullmask();

  // Bind the main thread to the CPU affinity of the current CUDA device, for the case when it is
  // used as a worker thread.
  utility::set_thread_affinity(
    device_properties::instance().get<device_properties::property::cpuAffinity>());

  // Get the current CUDA device to propagate to worker threads
  auto current_device = rmm::get_current_cuda_device();

  for (auto const& tasks_current_stage : task_graph_to_execute->stage_root_tasks) {
    auto const num_tasks_current_stage = tasks_current_stage.size();

    std::size_t num_workers =
      std::min(ctx_ref._query_context->parameters.num_workers, num_tasks_current_stage);
    if (num_tasks_current_stage) {
      auto root_task = utility::lock_or_throw(tasks_current_stage.front());
      GQE_LOG_TRACE("Execute stage {} with {} workers.", root_task->stage_id(), num_workers);
    } else {
      GQE_LOG_WARN("No tasks in the current stage (cannot identify the stage).");
      continue;
    }

    utility::nvtx_scoped_range stage_range(
      "Stage " + std::to_string(utility::lock_or_throw(tasks_current_stage.front())->stage_id()));

    if (num_workers == 1) {
      // If the number of worker threads is 1, we could avoid the thread spawning cost by using the
      // main thread.
      for (auto& task : tasks_current_stage) {
        // root task of a pipeline in a stage cannot belong to more than one pipeline
        auto curr_task = utility::lock_or_throw(task);
        assert(curr_task->pipeline_ids().size() == 1);
        auto const root_task_pipeline_id = *(curr_task->pipeline_ids().cbegin());
        GQE_LOG_TRACE(
          "Executing pipeline {} from stage {}.", root_task_pipeline_id, curr_task->stage_id());
        curr_task->execute();
      }
      cudf::get_default_stream().synchronize();
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers);
      std::vector<std::exception_ptr> worker_exceptions(num_workers, nullptr);

      for (std::size_t worker_idx = 0; worker_idx < num_workers; worker_idx++) {
        auto& worker_exception = worker_exceptions[worker_idx];

        workers.emplace_back([=, &tasks_current_stage, &worker_exception]() {
          try {
            // This is a sanity check to ensure that workers set the same CUDA device as the main
            // thread. The default is 0, but it can be changed elsewhere in the program (e.g.,
            // gqe-python).
            //
            // Reference:
            // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-visible-devices
            // https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs
            GQE_CUDA_TRY(cudaSetDevice(current_device.value()));
            // Bind the worker thread to the CPU affinity of the current CUDA device.
            utility::set_thread_affinity(
              device_properties::instance().get<device_properties::property::cpuAffinity>());
            for (std::size_t task_idx = worker_idx; task_idx < num_tasks_current_stage;
                 task_idx += num_workers) {
              auto curr_task = utility::lock_or_throw(tasks_current_stage[task_idx]);
              assert(curr_task->pipeline_ids().size() == 1);

              // root task of a pipeline in a stage cannot belong to more than one pipeline
              auto const root_task_pipeline_id = *(curr_task->pipeline_ids().cbegin());
              GQE_LOG_TRACE("Executing pipeline {} from stage {}.",
                            root_task_pipeline_id,
                            curr_task->stage_id());
              curr_task->execute();
            }
            cudf::get_default_stream().synchronize();
          } catch (const std::exception& e) {
            worker_exception = std::current_exception();
            GQE_LOG_ERROR("Error on worker {}: {}", worker_idx, e.what());
          }
        });
      }

      for (auto& worker : workers)
        worker.join();

      // Handle exceptions thrown in the worker threads. The main thread
      // rethrows the first exception it encounters. The rethrown exception can
      // be caught by the external function calling GQE.
      //
      // All workers _must_ be joined before rethrowing an exception to avoid a
      // "terminate called without an active exception" error!
      for (auto& exception : worker_exceptions) {
        if (exception != nullptr) { std::rethrow_exception(exception); }
      }
    }
  }
}

void execute_task_graph_multi_process(context_reference ctx_ref, task_graph const* task_graph)
{
  // Map from stage id to tasks in the stage
  std::unordered_map<int32_t, std::vector<std::shared_ptr<task>>> stage_map;

  // Map from stage id to set of parent stage ids
  std::unordered_map<int32_t, std::unordered_set<int32_t>> stage_dependency_map;

  for (auto const& tasks_current_stage : task_graph->stage_root_tasks) {
    for (auto const& task : tasks_current_stage) {
      auto curr_task = utility::lock_or_throw(task);
      stage_map[curr_task->stage_id()].push_back(curr_task);
      auto curr_stage_id = curr_task->stage_id();

      std::stack<gqe::task*> tasks;
      for (auto dependency : curr_task->dependencies()) {
        tasks.push(dependency);
      }
      while (!tasks.empty()) {
        auto curr_task = tasks.top();
        tasks.pop();
        for (auto dependency : curr_task->dependencies()) {
          if (dependency->stage_id() == curr_stage_id) {
            tasks.push(dependency);
          } else {
            stage_dependency_map[dependency->stage_id()].insert(curr_stage_id);
          }
        }
      }
    }
  }

  assert(ctx_ref._task_manager_context != nullptr);
  assert(ctx_ref._query_context != nullptr);
  assert(task_graph != nullptr);

  auto task_manager_context =
    dynamic_cast<multi_process_task_manager_context*>(ctx_ref._task_manager_context);
  if (!task_manager_context) {
    throw std::runtime_error(
      "execute_multi_process should be called with a multi-process task manager context");
  }

  auto execute_locally = [task_manager_context](task* t) {
    if (t->is_execute_on_all_ranks()) { return true; }
    auto execution_ranks = task_manager_context->scheduler->get_execution_ranks(t);
    if (execution_ranks.find(task_manager_context->comm->rank()) != execution_ranks.end()) {
      return true;
    }
    return false;
  };

  auto get_stage_id = [](std::weak_ptr<task> task) {
    return utility::lock_or_throw(task)->stage_id();
  };

  for (auto const& tasks_current_stage : task_graph->stage_root_tasks) {
    auto const num_tasks_current_stage = tasks_current_stage.size();

    std::size_t num_workers =
      std::min(ctx_ref._query_context->parameters.num_workers, num_tasks_current_stage);
    if (num_tasks_current_stage) {
      GQE_LOG_TRACE("Rank {}: Execute stage {} with {} workers.",
                    task_manager_context->comm->rank(),
                    get_stage_id(tasks_current_stage.front()),
                    num_workers);
    } else {
      GQE_LOG_WARN("Rank {}: No tasks in the current stage (cannot identify the stage).",
                   task_manager_context->comm->rank());
      continue;
    }

    utility::nvtx_scoped_range stage_range(
      "Stage " + std::to_string(get_stage_id(tasks_current_stage.front())));

    if (num_workers == 1) {
      // If the number of worker threads is 1, we could avoid the thread spawning cost by using the
      // main thread.
      std::exception_ptr stage_exception = nullptr;
      try {
        // Bind the main thread to the CPU affinity of the current CUDA device.
        utility::set_thread_affinity(
          device_properties::instance().get<device_properties::property::cpuAffinity>());
        for (auto& task : tasks_current_stage) {
          // root task of a pipeline in a stage cannot belong to more than one pipeline
          auto curr_task = utility::lock_or_throw(task);
          assert(curr_task->pipeline_ids().size() == 1);
          auto const root_task_pipeline_id = *(curr_task->pipeline_ids().cbegin());
          if (execute_locally(curr_task.get())) {
            GQE_LOG_TRACE("Rank {}: Executing pipeline {} from stage {}.",
                          task_manager_context->comm->rank(),
                          root_task_pipeline_id,
                          curr_task->stage_id());
            task_manager_context->migration_service->register_task(curr_task.get());
            curr_task->execute();
          } else {
            GQE_LOG_TRACE("Rank {}: Pipeline {} from stage {} is not executed locally.",
                          task_manager_context->comm->rank(),
                          root_task_pipeline_id,
                          curr_task->stage_id());
          }
        }
        cudf::get_default_stream().synchronize();  // ensure completion of launched GPU activity
      } catch (const std::exception&) {
        stage_exception = std::current_exception();
      }

      task_manager_context->comm->propagate_error(stage_exception);

      task_manager_context->comm
        ->barrier_world();  // ensure all ranks have completed the current stage
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_workers);
      std::vector<std::exception_ptr> worker_exceptions(num_workers, nullptr);
      std::vector<task*> tasks_to_execute;
      for (auto& task : tasks_current_stage) {
        auto curr_task = utility::lock_or_throw(task);
        if (execute_locally(curr_task.get())) {
          task_manager_context->migration_service->register_task(curr_task.get());
          tasks_to_execute.push_back(curr_task.get());
        }
      }

      for (std::size_t worker_idx = 0; worker_idx < num_workers; worker_idx++) {
        auto& worker_exception = worker_exceptions[worker_idx];

        workers.emplace_back([=, &tasks_to_execute, &worker_exception]() {
          try {
            GQE_CUDA_TRY(cudaSetDevice(task_manager_context->comm->device_id().value()));
            // Bind the worker thread to the CPU affinity of the current CUDA device.
            utility::set_thread_affinity(
              device_properties::instance().get<device_properties::property::cpuAffinity>());
            for (std::size_t task_idx = worker_idx; task_idx < tasks_to_execute.size();
                 task_idx += num_workers) {
              auto curr_task = tasks_to_execute[task_idx];
              assert(curr_task->pipeline_ids().size() == 1);

              // root task of a pipeline in a stage cannot belong to more than one pipeline
              auto const root_task_pipeline_id = *(curr_task->pipeline_ids().cbegin());
              GQE_LOG_TRACE("Rank {}: Executing pipeline {} from stage {}.",
                            task_manager_context->comm->rank(),
                            root_task_pipeline_id,
                            curr_task->stage_id());
              curr_task->execute();
            }
            cudf::get_default_stream().synchronize();
          } catch (const std::exception&) {
            worker_exception = std::current_exception();
          }
        });
      }

      for (auto& worker : workers)
        worker.join();

      // Determine if this rank encountered an exception and propagate across ranks
      std::exception_ptr stage_exception = nullptr;
      for (auto& exception : worker_exceptions) {
        if (exception != nullptr) {
          stage_exception = exception;
          break;
        }
      }

      task_manager_context->comm->propagate_error(stage_exception);
      task_manager_context->comm
        ->barrier_world();  // ensure all ranks have completed the current stage
    }

    // Release dependencies for current stage tasks. Tasks in the current stage may be executed on
    // by another process, we need to remove dependencies to prevent dangling resources.
    // Note: tasks are NOT unregistered here because a later stage on another rank may still need
    // to fetch their metadata via RPC (migrate_result). Cleanup happens when the query finishes
    // via query_registration_guard in plan_executor.
    for (auto& task : tasks_current_stage) {
      auto curr_task = utility::lock_or_throw(task);
      curr_task->remove_dependencies();
    }

    // Release child stages
    for (auto& [child_stage_id, parent_stage_ids] : stage_dependency_map) {
      if (!tasks_current_stage.empty()) {
        parent_stage_ids.erase(get_stage_id(
          tasks_current_stage
            .front()));  // parent stage is completed, remove it from the dependency map
      }
      if (parent_stage_ids.empty()) {
        stage_map.erase(child_stage_id);  // child has no more parents, we can release it
      }
    }
  }
}

// FIXME: Current implementation of task_graph_builder::generate_task_graph_visitor::visit cannot
// assign the parent task and the child task to the same stage if the child task has already been
// generated with a different stage and is retrieved from the task cache
// (`task_graph_builder::generate_task_graph_visitor::is_cached`), even if the edge is not a
// pipeline breaker.

void task_graph_builder::generate_task_graph_visitor::visit(physical::read_relation* relation)
{
  if (is_cached(relation)) return;

  auto const table_name     = relation->table_name();
  auto const column_names   = relation->column_names();
  auto const partial_filter = relation->partial_filter_unsafe();
  std::vector<std::shared_ptr<task>> concatenated_subquery_task;

  if (partial_filter && (partial_filter->type() == expression::expression_type::subquery)) {
    // If it's a subquery, we know it's an in-predicate
    auto const subquery = dynamic_cast<in_predicate_expression* const>(partial_filter);
    // Make sure dynamic_cast was successful
    assert(subquery != nullptr);
    auto const subquery_relations = relation->subqueries_unsafe();

    auto const subquery_tasks =
      _builder->generate_tasks(subquery_relations[subquery->relation_index()]);
    concatenated_subquery_task = {_builder->concatenate(subquery_tasks)};
  }

  auto const data_types = relation->output_data_types();

  auto tbl = _builder->_table_provider->get_table(relation->table_name());
  if (!tbl || !tbl->is_readable()) {
    throw std::logic_error("read_relation for table \"" + table_name +
                           "\" has no readable storage");
  }

  auto readable_view = tbl->readable_view();

  // An empty table reports zero concurrent readers (e.g. an in-memory table with no row groups).
  // Always create at least one read partition so the read yields an empty result rather than
  // producing zero task parameters, which would divide by zero in get_read_tasks.
  const auto npartitions =
    std::max(1,
             std::min(tbl->max_concurrent_readers(),
                      _builder->_ctx_ref._query_context->parameters.num_partitions));

  std::vector<storage::readable_view::task_parameters> task_parameters(npartitions);
  std::generate(task_parameters.begin(),
                task_parameters.end(),
                [&]() -> storage::readable_view::task_parameters {
                  std::vector<std::shared_ptr<gqe::task>> subqueries;
                  return {_builder->_current_task_id++,
                          partial_filter ? partial_filter->clone() : nullptr,
                          concatenated_subquery_task};
                });

  auto tasks = readable_view->get_read_tasks(std::move(task_parameters),
                                             _builder->_ctx_ref,
                                             _builder->_current_stage_id,
                                             std::move(column_names),
                                             std::move(data_types));

  _generated_tasks.reserve(_generated_tasks.size() + tasks.size());
  std::move(tasks.begin(), tasks.end(), std::back_inserter(_generated_tasks));

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::write_relation* relation)
{
  if (is_cached(relation)) return;

  auto const column_names = relation->column_names();
  auto const data_types   = relation->output_data_types();

  auto tbl = _builder->_table_provider->get_table(relation->table_name());
  if (!tbl || !tbl->is_writeable()) {
    throw std::logic_error("write_relation for table \"" + relation->table_name() +
                           "\" has no writeable storage");
  }

  auto const children = relation->children_unsafe();
  assert(children.size() == 1);

  // FIXME: This implementation requires that the number of concurrent writes
  // matches the number of child tasks. In future, we should handle mismatches
  // better. For example, we could concatenate child tasks to the exact number
  // of writers, or dynamically rescale the writer's concurrency (e.g., Parquet
  // pages).
  auto input_tasks = tbl->max_concurrent_writers() == 1
                       ? std::vector{_builder->concatenate(_builder->generate_tasks(children[0]))}
                       : _builder->generate_tasks(children[0]);

  auto writeable_view = tbl->writeable_view();

  std::vector<storage::writeable_view::task_parameters> task_parameters(input_tasks.size());
  std::generate(
    task_parameters.begin(),
    task_parameters.end(),
    [this, it = input_tasks.begin()]() mutable -> storage::writeable_view::task_parameters {
      return {_builder->_current_task_id++, std::move(*it++)};
    });

  if (!_builder->_write_statistics) {
    _builder->_write_statistics =
      std::make_unique<table_statistics_manager>(gqe::table_statistics(0, data_types));
  }

  auto tasks = writeable_view->get_write_tasks(std::move(task_parameters),
                                               _builder->_ctx_ref,
                                               _builder->_current_stage_id,
                                               std::move(column_names),
                                               std::move(data_types),
                                               _builder->_write_statistics.get());

  _generated_tasks.reserve(_generated_tasks.size() + tasks.size());
  std::move(tasks.begin(), tasks.end(), std::back_inserter(_generated_tasks));

  update_cache(relation);
}

namespace {

// flowcsd spill mode: the equality keys of a join condition, parsed as join_task parses them
// (join.cpp parse_join_condition): the operands of EQUAL / NULL_EQUALS under AND.
void collect_equality_keys(expression const* condition,
                           std::vector<expression const*>& left_keys,
                           std::vector<expression const*>& right_keys)
{
  if (condition->type() != expression::expression_type::binary_op) { return; }
  auto const* binary_op = dynamic_cast<binary_op_expression const*>(condition);
  auto const operands   = condition->children();
  switch (binary_op->binary_operator()) {
    case cudf::binary_operator::EQUAL:
    case cudf::binary_operator::NULL_EQUALS:
      left_keys.push_back(operands[0]);
      right_keys.push_back(operands[1]);
      break;
    case cudf::binary_operator::LOGICAL_AND:
    case cudf::binary_operator::NULL_LOGICAL_AND:
      collect_equality_keys(operands[0], left_keys, right_keys);
      collect_equality_keys(operands[1], left_keys, right_keys);
      break;
    default: break;
  }
}

// flowcsd spill mode: number of Parquet files read under `relation` (the planner has no row
// counts, catalog.cpp:113-117, so files stand in for size), and the names of the tables read (a
// table read twice is named twice).
int64_t count_input_files(physical::relation* relation,
                          storage::table_provider const* provider,
                          std::vector<std::string>& table_names)
{
  int64_t files = 0;
  if (auto* read = dynamic_cast<physical::read_relation*>(relation)) {
    files += provider->get_table(read->table_name())->max_concurrent_readers();
    table_names.push_back(read->table_name());
  }
  for (auto* child : relation->children_unsafe()) {
    files += count_input_files(child, provider, table_names);
  }
  return files;
}

// flowcsd spill mode: names joined by `separator` in their order.
std::string join_names(std::vector<std::string> const& names, char const* separator)
{
  std::string joined;
  for (auto const& name : names) {
    if (!joined.empty()) { joined += separator; }
    joined += name;
  }
  return joined;
}

// flowcsd spill mode: whether a join's build input is filtered: its subtree has a filter, a read
// with a pushed-down partial filter, or a left semi or anti join. An unfiltered TPC-H build side
// holds the key of every probe row (foreign keys), so a probe-side filter would drop nothing.
bool build_is_filtered(physical::relation* relation)
{
  if (dynamic_cast<physical::filter_relation*>(relation) != nullptr) { return true; }
  if (auto* read = dynamic_cast<physical::read_relation*>(relation)) {
    if (read->partial_filter_unsafe() != nullptr) { return true; }
  }
  if (auto* join = dynamic_cast<physical::join_relation_base*>(relation)) {
    auto const join_type = join->join_type();
    if (join_type == join_type_type::left_semi || join_type == join_type_type::left_anti) {
      return true;
    }
  }
  auto const children = relation->children_unsafe();
  return std::any_of(children.begin(), children.end(), build_is_filtered);
}

// flowcsd spill mode: the column positions an expression reads.
void collect_column_references(expression const* expr, std::set<cudf::size_type>& columns)
{
  if (expr->type() == expression::expression_type::column_reference) {
    columns.insert(static_cast<column_reference_expression const*>(expr)->column_idx());
  }
  for (auto const* child : expr->children()) {
    collect_column_references(child, columns);
  }
}

char const* join_type_name(join_type_type join_type)
{
  switch (join_type) {
    case join_type_type::inner: return "inner";
    case join_type_type::left: return "left";
    case join_type_type::left_semi: return "left_semi";
    case join_type_type::left_anti: return "left_anti";
    default: return "other";
  }
}

}  // namespace

std::vector<std::shared_ptr<task>> task_graph_builder::generate_spill_partition_tasks(
  std::vector<std::shared_ptr<task>> input_tasks,
  std::vector<expression const*> const& key_exprs,
  cudf::size_type key_column_offset,
  std::shared_ptr<spill_set> const& spill,
  int32_t side)
{
  std::vector<std::shared_ptr<task>> partition_tasks;
  partition_tasks.reserve(input_tasks.size());
  for (auto& input_task : input_tasks) {
    partition_tasks.push_back(std::make_shared<spill_partition_task>(_ctx_ref,
                                                                     _current_task_id,
                                                                     _current_stage_id,
                                                                     std::move(input_task),
                                                                     clone_expressions(key_exprs),
                                                                     key_column_offset,
                                                                     spill,
                                                                     side));
    _current_task_id++;
  }
  insert_pipeline_breaker(partition_tasks);
  return partition_tasks;
}

std::string task_graph_builder::next_spill_signature(std::string const& kind,
                                                     std::vector<std::string> tables,
                                                     std::size_t keys)
{
  // No build side in it: the planner picks the build side from file counts, which differ between
  // scale factors.
  std::sort(tables.begin(), tables.end());
  auto const name    = std::format("{}({}):keys={}", kind, join_names(tables, "+"), keys);
  auto const ordinal = _spill_signature_counts[name]++;
  return std::format("{}:{}", name, ordinal);
}

bool task_graph_builder::generate_task_graph_visitor::generate_spilled_join(
  physical::broadcast_join_relation* relation)
{
  auto const children = relation->children_unsafe();
  std::vector<expression const*> left_keys;
  std::vector<expression const*> right_keys;
  collect_equality_keys(relation->condition(), left_keys, right_keys);

  // Which input a broadcast join would hold whole in GPU memory (build) and which it streams
  // (probe).
  bool const build_is_right = relation->policy() == physical::broadcast_policy::right;
  int32_t const build_side  = build_is_right ? 1 : 0;
  int32_t const probe_side  = 1 - build_side;
  auto const join_type      = relation->join_type();
  std::vector<std::string> build_tables;
  std::vector<std::string> probe_tables;
  auto const build_files =
    count_input_files(children[build_side], _builder->_table_provider, build_tables);
  auto const probe_files =
    count_input_files(children[probe_side], _builder->_table_provider, probe_tables);
  // Filter conditions inside a join come only from hand-written plans (plan_join.cpp:155-156), and
  // join_task accepts them only together with the hash-map cache, which a Grace join cannot share.
  bool const has_filter_condition =
    relation->left_filter_condition() != nullptr || relation->right_filter_condition() != nullptr;
  // A left join that builds on its left input would emit its unmatched build rows once per probe
  // task in stage B; it keeps the upstream path (GQE's planner does not build such a plan).
  bool const left_join_builds_left = join_type == join_type_type::left && build_side == 0;
  // Every other equality join goes through the hybrid path; how much of it spills is decided at run
  // time from the actual bytes (a join whose build side fits spills nothing).
  bool const hybrid = !left_keys.empty() && !has_filter_condition && !left_join_builds_left;
  // An inner join, and a left semi join that builds on the right, drop every probe row without a
  // matching build key. With a filtered build side they get a probe-side filter that drops those
  // rows before partitioning (spill_filter.hpp); an unfiltered build side holds every probe key.
  bool const drops_unmatched_probe_rows =
    join_type == join_type_type::inner ||
    (join_type == join_type_type::left_semi && build_side == 1);
  bool const filtered_build = build_is_filtered(children[build_side]);
  bool const filters_probe  = drops_unmatched_probe_rows && filtered_build;
  // The columns the join reads neither in its condition nor in its output are left out of written
  // slices (spill_operator_plan::dead_columns); right indices are offset by the left column count.
  auto const left_num_columns =
    static_cast<cudf::size_type>(children[0]->output_data_types().size());
  auto const right_num_columns =
    static_cast<cudf::size_type>(children[1]->output_data_types().size());
  std::set<cudf::size_type> used_columns;
  collect_column_references(relation->condition(), used_columns);
  for (auto const column : relation->projection_indices()) {
    used_columns.insert(column);
  }
  std::array<std::vector<cudf::size_type>, 2> dead_columns;
  for (cudf::size_type column = 0; column < left_num_columns; column++) {
    if (used_columns.count(column) == 0) { dead_columns[0].push_back(column); }
  }
  for (cudf::size_type column = 0; column < right_num_columns; column++) {
    if (used_columns.count(left_num_columns + column) == 0) { dead_columns[1].push_back(column); }
  }
  auto const description = std::format(
    "join type={} build_side={} build_tables={} build_files={} probe_tables={} probe_files={} "
    "equality_keys={} left_columns={} left_unused={} right_columns={} right_unused={} "
    "filtered_build={}",
    join_type_name(join_type),
    build_is_right ? "right" : "left",
    join_names(build_tables, ","),
    build_files,
    join_names(probe_tables, ","),
    probe_files,
    left_keys.size(),
    left_num_columns,
    dead_columns[0].size(),
    right_num_columns,
    dead_columns[1].size(),
    filtered_build);
  if (!hybrid) {
    GQE_LOG_INFO("flowcsd_spill_plan {} hybrid=false{}",
                 description,
                 left_join_builds_left ? " hybrid_off=left_join_builds_left" : "");
    return false;
  }
  auto tables = build_tables;
  tables.insert(tables.end(), probe_tables.begin(), probe_tables.end());
  auto const signature = _builder->next_spill_signature(
    std::string(join_type_name(join_type)) + "_join", std::move(tables), left_keys.size());
  auto const operator_partition_count = spill_partitions_for(signature);
  auto const partitions               = operator_partition_count.partitions;
  GQE_LOG_INFO("flowcsd_spill_plan {} hybrid=true signature={} partitions={} partitions_from={}",
               description,
               signature,
               partitions,
               operator_partition_count.from_table ? "table" : "default");
  count_hybrid_operator(hybrid_operator::join);

  // A semi or anti join that builds on its left input must see every probe row of a key at once
  // (the broadcast path merges position lists across probe tasks), so its probe rows cannot be
  // joined per probe task: it runs as a Grace join with every partition on SSD.
  bool const semi_or_anti =
    join_type == join_type_type::left_semi || join_type == join_type_type::left_anti;
  bool const streams_probe = !(semi_or_anti && build_side == 0);
  auto const& parameters   = _builder->_ctx_ref._query_context->parameters;
  // Stages B and C share one stage: C's reads rely on B's pipelines having run first, in order.
  GQE_EXPECTS(parameters.num_workers == 1, "spill mode needs GQE_NUM_WORKERS=1");
  auto const pool_bytes =
    parameters.max_query_memory.value_or(rmm::available_device_memory().second);
  // The build input comes first: the number of its tasks projects k and sizes the filter.
  auto build_input_tasks = _builder->generate_tasks(children[build_side]);
  auto const build_tasks = static_cast<int64_t>(build_input_tasks.size());
  spill_operator_plan plan;
  plan.num_sides                = 2;
  plan.description              = description;
  plan.signature                = signature;
  plan.partitions               = partitions;
  plan.held_side                = streams_probe ? build_side : -1;
  plan.probe_side               = probe_side;
  plan.held_input_tasks         = build_tasks;
  plan.probe_filter_build_tasks = filters_probe ? build_tasks : int64_t{0};
  plan.dead_columns             = dead_columns;
  plan.resident_budget_bytes    = spill_resident_budget_bytes(pool_bytes);
  plan.pool_bytes               = pool_bytes;
  auto spill                    = make_spill_set(std::move(plan));
  // The right keys reference the concatenation of both inputs (join.cpp:308-312).
  std::array<std::vector<expression const*>, 2> const keys{left_keys, right_keys};
  std::array<cudf::size_type, 2> const key_column_offsets{0, left_num_columns};
  gqe::unique_keys_policy const unique_keys_pol =
    parameters.join_use_unique_keys ? relation->unique_keys_policy() : gqe::unique_keys_policy::none;
  // Join tasks take (left, right); put the build and probe inputs on the join's own sides. Perfect
  // hashing stays off: libperfect sizes its table by the significant key bits
  // (xor_hash_table.cuh:773-786), which every hash partition still spans, so the table would not
  // shrink with the partition (2^33 slots x 4 B for l_orderkey at SF1000).
  auto const generate_join = [&](std::shared_ptr<task> build_input,
                                 std::shared_ptr<task> probe_input,
                                 std::shared_ptr<join_hash_map_cache> hash_map_cache) {
    auto left_input  = build_side == 0 ? std::move(build_input) : std::move(probe_input);
    auto right_input = build_side == 0 ? std::move(probe_input) : std::move(build_input);
    auto joined      = std::make_shared<join_task>(_builder->_ctx_ref,
                                              _builder->_current_task_id,
                                              _builder->_current_stage_id,
                                              std::move(left_input),
                                              std::move(right_input),
                                              join_type,
                                              relation->condition()->clone(),
                                              relation->projection_indices(),
                                              std::move(hash_map_cache),
                                              /*materialize_output=*/true,
                                              unique_keys_pol,
                                              /*perfect_hashing=*/false,
                                              /*left_filter_condition=*/nullptr,
                                              /*right_filter_condition=*/nullptr,
                                              /*mark_join=*/false);
    _builder->_current_task_id++;
    return joined;
  };

  // Stage A: build input -> build partitions, resident while they fit the budget, else on SSD.
  auto build_partitions = _builder->generate_spill_partition_tasks(std::move(build_input_tasks),
                                                                  keys[build_side],
                                                                  key_column_offsets[build_side],
                                                                  spill,
                                                                  build_side);
  auto probe_input_tasks = _builder->generate_tasks(children[probe_side]);
  std::vector<std::shared_ptr<task>> probe_partitions;  // Grace path only
  if (streams_probe) {
    // Stage B: split each probe input by partition. Rows of resident partitions join the resident
    // build at once (one join per probe input, like the broadcast join; inner and left joins share
    // one hash table); rows of spilled partitions go to SSD.
    auto resident_build = std::make_shared<spill_resident_build_task>(_builder->_ctx_ref,
                                                                      _builder->_current_task_id,
                                                                      _builder->_current_stage_id,
                                                                      build_partitions,
                                                                      spill,
                                                                      build_side);
    _builder->_current_task_id++;
    std::shared_ptr<join_hash_map_cache> hash_map_cache;
    if (join_type == join_type_type::inner || join_type == join_type_type::left) {
      hash_map_cache = std::make_shared<join_hash_map_cache>(
        build_side == 1 ? join_hash_map_cache::build_location::right
                        : join_hash_map_cache::build_location::left);
    }
    for (auto& probe_input : probe_input_tasks) {
      auto split = std::make_shared<spill_probe_split_task>(_builder->_ctx_ref,
                                                            _builder->_current_task_id,
                                                            _builder->_current_stage_id,
                                                            std::move(probe_input),
                                                            clone_expressions(keys[probe_side]),
                                                            key_column_offsets[probe_side],
                                                            spill,
                                                            probe_side);
      _builder->_current_task_id++;
      _generated_tasks.push_back(generate_join(resident_build, std::move(split), hash_map_cache));
    }
  } else {
    probe_partitions = _builder->generate_spill_partition_tasks(std::move(probe_input_tasks),
                                                               keys[probe_side],
                                                               key_column_offsets[probe_side],
                                                               spill,
                                                               probe_side);
  }

  // Stage C (the same stage as B, after it): one join per partition from its spilled data, like
  // the shuffle join (visit(shuffle_join_relation)): no shared hash-map cache, no mark join.
  // Partitions without spill data (joined in stage B) do not run (spill_partition_done_task). How a
  // partition join's output rows tell how many of its spilled probe rows matched (logged per
  // operator; spill.hpp spill_match_rule):
  bool const build_keys_unique =
    unique_keys_pol ==
    (build_side == 0 ? gqe::unique_keys_policy::left : gqe::unique_keys_policy::right);
  auto match_rule = spill_match_rule::unknown;
  if (join_type == join_type_type::inner && build_keys_unique) {
    match_rule = spill_match_rule::output_rows;
  } else if (join_type == join_type_type::left_semi && build_side == 1) {
    match_rule = spill_match_rule::output_rows;
  } else if (join_type == join_type_type::left_anti && build_side == 1) {
    match_rule = spill_match_rule::probe_rows_minus_output;
  }
  auto const output_types = relation->output_data_types();
  for (int32_t partition = 0; partition < partitions; partition++) {
    auto build_read = std::make_shared<spill_read_task>(_builder->_ctx_ref,
                                                        _builder->_current_task_id,
                                                        _builder->_current_stage_id,
                                                        build_partitions,
                                                        spill,
                                                        build_side,
                                                        partition);
    _builder->_current_task_id++;
    auto probe_read = std::make_shared<spill_read_task>(_builder->_ctx_ref,
                                                        _builder->_current_task_id,
                                                        _builder->_current_stage_id,
                                                        probe_partitions,
                                                        spill,
                                                        probe_side,
                                                        partition);
    _builder->_current_task_id++;
    auto partition_join = generate_join(std::move(build_read), std::move(probe_read), nullptr);
    _generated_tasks.push_back(std::make_shared<spill_partition_done_task>(_builder->_ctx_ref,
                                                                           _builder->_current_task_id,
                                                                           _builder->_current_stage_id,
                                                                           std::move(partition_join),
                                                                           spill,
                                                                           partition,
                                                                           match_rule,
                                                                           output_types));
    _builder->_current_task_id++;
  }
  return true;
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::broadcast_join_relation* relation)
{
  if (is_cached(relation)) return;

  auto const relation_join_type = relation->join_type();
  if (relation_join_type == join_type_type::full || relation_join_type == join_type_type::single)
    throw std::logic_error("Broadcast join does not support full join or single join");

  auto const children = relation->children_unsafe();

  // flowcsd spill mode: a hybrid hash join (resident and SSD partitions) instead of a broadcast join.
  if (get_spill_settings().partitions > 0 && generate_spilled_join(relation)) {
    update_cache(relation);
    return;
  }

  bool cache_enabled = _builder->_ctx_ref._query_context->parameters.join_use_hash_map_cache;

  // only physical_plan_builder::plan_join code path sets use_hash_map_cache() true, so we
  // can just honor it here without disrupting the gqe-python's legacy code for handcoded plans
  if (relation->use_hash_map_cache()) { cache_enabled = true; }

  std::shared_ptr<gqe::join_hash_map_cache> hash_map_cache = nullptr;

  gqe::unique_keys_policy unique_keys_pol =
    _builder->_ctx_ref._query_context->parameters.join_use_unique_keys
      ? relation->unique_keys_policy()
      : gqe::unique_keys_policy::none;
  bool perfect_hashing = _builder->_ctx_ref._query_context->parameters.join_use_perfect_hash &&
                         relation->perfect_hashing();

  if (relation->policy() == physical::broadcast_policy::right) {
    // current mark join cannot semantically handle broadcast-right
    constexpr bool mark_join = false;

    // Generate the right children tasks
    auto right_tasks = _builder->generate_tasks(children[1]);

    // Insert a pipeline breaker since the right tasks need to be broadcasted
    _builder->insert_pipeline_breaker(right_tasks);

    // Generate the left children tasks
    auto left_tasks = _builder->generate_tasks(children[0]);

    // Concatenate the right child tasks
    // Note that we don't insert a pipeline breaker here so that the join tasks have the same
    // stage as the concatenate task. This way, the concatenate task implements the broadcasting
    // instead of being executed on a single GPU.
    auto concatenated_right_task = _builder->concatenate(std::move(right_tasks), false);

    if (cache_enabled) {
      hash_map_cache =
        std::make_shared<gqe::join_hash_map_cache>(gqe::join_hash_map_cache::build_location::right);
    }

    // Generate the join tasks
    for (auto& left_task : left_tasks) {
      _generated_tasks.push_back(std::make_shared<join_task>(
        _builder->_ctx_ref,
        _builder->_current_task_id,
        _builder->_current_stage_id,
        std::move(left_task),
        concatenated_right_task,
        relation_join_type,
        relation->condition()->clone(),
        relation->projection_indices(),
        hash_map_cache,
        true,
        unique_keys_pol,
        perfect_hashing,
        relation->left_filter_condition() ? relation->left_filter_condition()->clone() : nullptr,
        relation->right_filter_condition() ? relation->right_filter_condition()->clone() : nullptr,
        mark_join));
      _builder->_current_task_id++;
    }
  } else {
    if (relation_join_type != join_type_type::inner &&
        relation_join_type != join_type_type::left_semi &&
        relation_join_type != join_type_type::left_anti) {
      throw std::logic_error(
        "Broadcast join does not support left broadcast for this join; only inner/semi/anti join "
        "can broadcast the left table.");
    }
    const bool mark_join = _builder->_ctx_ref._query_context->parameters.join_use_mark_join;
    // TODO: we can delete the code below when gqe-python's legacy code is removed
    // We enable cache ALWAYS with mark_join for best performance.
    // TODO: this can be removed when perfect hashing and hash map caching are mutually compatible
    // See https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/161
    if (mark_join && (relation_join_type == join_type_type::left_semi ||
                      relation_join_type == join_type_type::left_anti)) {
      cache_enabled = true;
    }

    // Generate the left children tasks
    auto left_tasks = _builder->generate_tasks(children[0]);

    // Insert a pipeline breaker since the left tasks need to be broadcasted
    _builder->insert_pipeline_breaker(left_tasks);

    // Generate the right children tasks
    auto right_tasks = _builder->generate_tasks(children[1]);

    // Concatenate the left child tasks
    auto concatenated_left_task = _builder->concatenate(std::move(left_tasks), false);

    if (cache_enabled) {
      hash_map_cache =
        std::make_shared<gqe::join_hash_map_cache>(gqe::join_hash_map_cache::build_location::left);
    }

    bool separate_materialization = false;
    if (relation_join_type == join_type_type::left_semi ||
        relation_join_type == join_type_type::left_anti) {
      // We do not materialize the join result in the join task itself for the left-semi and
      // left-anti join because each join task only produces part of the position list. Instead, the
      // position lists will be merged together and the join will be materialized in a single
      // `materialize_join_from_position_lists_task`.
      separate_materialization = true;
    }

    // Generate the join tasks
    std::vector<std::shared_ptr<task>> join_tasks;
    for (auto& right_task : right_tasks) {
      join_tasks.push_back(std::make_shared<join_task>(
        _builder->_ctx_ref,
        _builder->_current_task_id,
        _builder->_current_stage_id,
        concatenated_left_task,
        std::move(right_task),
        relation_join_type,
        relation->condition()->clone(),
        relation->projection_indices(),
        hash_map_cache,
        !separate_materialization,
        unique_keys_pol,
        perfect_hashing,
        relation->left_filter_condition() ? relation->left_filter_condition()->clone() : nullptr,
        relation->right_filter_condition() ? relation->right_filter_condition()->clone() : nullptr,
        mark_join));
      _builder->_current_task_id++;
    }

    // Generate a separate `materialize_join_from_position_lists_task` for semi/anti join
    if (separate_materialization) {
      _builder->insert_pipeline_breaker(join_tasks);
      std::vector<std::shared_ptr<gqe::task>> late_materialization_dependencies;

      if (mark_join && hash_map_cache) {
        auto task_manager_context = dynamic_cast<multi_process_task_manager_context*>(
          _builder->_ctx_ref._task_manager_context);
        bool is_multi_process = (task_manager_context != nullptr);
        // If not running in multi-process mode, set num_ranks to 1.
        auto num_ranks = is_multi_process ? task_manager_context->comm->world_size() : 1;
        // It would happen that join_tasks.size() < world_size.
        auto num_extract_tasks = std::min(static_cast<std::size_t>(num_ranks), join_tasks.size());
        std::vector<std::shared_ptr<task>> extract_mark_join_positions_tasks(num_extract_tasks);
        for (size_t i = 0; i < num_extract_tasks; i++) {
          extract_mark_join_positions_tasks[i] =
            std::make_shared<extract_mark_join_positions_task>(_builder->_ctx_ref,
                                                               _builder->_current_task_id,
                                                               _builder->_current_stage_id,
                                                               join_tasks,
                                                               relation_join_type,
                                                               hash_map_cache);
          _builder->_current_task_id++;
        }
        late_materialization_dependencies = std::move(extract_mark_join_positions_tasks);
        _builder->insert_pipeline_breaker(late_materialization_dependencies);
      } else {
        late_materialization_dependencies = std::move(join_tasks);
      }

      _generated_tasks.push_back(std::make_shared<materialize_join_from_position_lists_task>(
        _builder->_ctx_ref,
        _builder->_current_task_id,
        _builder->_current_stage_id,
        std::move(concatenated_left_task),
        std::move(late_materialization_dependencies),
        relation_join_type,
        relation->projection_indices()));

      _builder->_current_task_id++;
    } else {
      for (auto& join_task : join_tasks) {
        _generated_tasks.push_back(std::move(join_task));
      }
    }
  }

  update_cache(relation);
}

std::vector<std::shared_ptr<task>> task_graph_builder::generate_partition_tasks(
  physical::relation* child_relation, std::vector<expression*> const& exprs_to_hash)
{
  auto child_tasks = generate_tasks(child_relation);

  auto const num_shuffle_partitions = _ctx_ref._query_context->parameters.num_shuffle_partitions;

  std::vector<std::shared_ptr<task>> partition_tasks;

  for (auto& child_task : child_tasks) {
    partition_tasks.push_back(std::make_shared<partition_task>(_ctx_ref,
                                                               _current_task_id,
                                                               _current_stage_id,
                                                               child_task,
                                                               clone_expressions(exprs_to_hash),
                                                               num_shuffle_partitions));
    _current_task_id++;
  }

  insert_pipeline_breaker(partition_tasks);

  return partition_tasks;
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::shuffle_join_relation* relation)
{
  if (is_cached(relation)) return;
  GQE_LOG_TRACE("task_graph_builder::generate_task_graph_visitor::visit(shuffle_join_relation)");
  auto const relation_join_type = relation->join_type();
  if (relation_join_type == join_type_type::full || relation_join_type == join_type_type::single)
    throw std::logic_error("shuffle join does not support full join or single join");

  auto const children = relation->children_unsafe();

  // Generate the left children tasks
  auto left_tasks = _builder->generate_tasks(children[0]);
  // Generate the right children tasks
  auto right_tasks = _builder->generate_tasks(children[1]);

  gqe::unique_keys_policy unique_keys_pol =
    _builder->_ctx_ref._query_context->parameters.join_use_unique_keys
      ? relation->unique_keys_policy()
      : gqe::unique_keys_policy::none;
  bool perfect_hashing = _builder->_ctx_ref._query_context->parameters.join_use_perfect_hash &&
                         relation->perfect_hashing();

  // Generate the partition merge tasks and join tasks
  auto const num_shuffle_partitions =
    _builder->_ctx_ref._query_context->parameters.num_shuffle_partitions;
  // Make sure the physical plan is valid
  assert(dynamic_cast<partition_task*>(left_tasks[0].get()) != nullptr ||
         (int)left_tasks.size() == num_shuffle_partitions);
  assert(dynamic_cast<partition_task*>(right_tasks[0].get()) != nullptr ||
         (int)right_tasks.size() == num_shuffle_partitions);

  for (int32_t partition_idx = 0; partition_idx < num_shuffle_partitions; partition_idx++) {
    std::shared_ptr<task> left_input_task;
    std::shared_ptr<task> right_input_task;
    // check if the left_tasks are partition_task, we only need to check the first task
    if (dynamic_cast<partition_task*>(left_tasks[0].get())) {
      auto left_tasks_cloned = left_tasks;
      // Generate the partition merge tasks
      left_input_task = std::make_shared<partition_merge_task>(_builder->_ctx_ref,
                                                               _builder->_current_task_id,
                                                               _builder->_current_stage_id,
                                                               left_tasks_cloned,
                                                               partition_idx);
      _builder->_current_task_id++;
    } else {
      left_input_task = left_tasks[partition_idx];
    }
    // check if the right_tasks are partition_task, we only need to check the first task
    if (dynamic_cast<partition_task*>(right_tasks[0].get())) {
      auto right_tasks_cloned = right_tasks;
      right_input_task        = std::make_shared<partition_merge_task>(_builder->_ctx_ref,
                                                                _builder->_current_task_id,
                                                                _builder->_current_stage_id,
                                                                right_tasks_cloned,
                                                                partition_idx);

      _builder->_current_task_id++;
    } else {
      right_input_task = right_tasks[partition_idx];
    }
    // Generate the join tasks
    _generated_tasks.push_back(std::make_shared<join_task>(_builder->_ctx_ref,
                                                           _builder->_current_task_id,
                                                           _builder->_current_stage_id,
                                                           std::move(left_input_task),
                                                           std::move(right_input_task),
                                                           relation_join_type,
                                                           relation->condition()->clone(),
                                                           relation->projection_indices(),
                                                           /*hash_map_cache=*/nullptr,
                                                           /*materialize_output=*/true,
                                                           unique_keys_pol,
                                                           perfect_hashing));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::shuffle_relation* relation)
{
  if (is_cached(relation)) return;

  GQE_LOG_TRACE("task_graph_builder::generate_task_graph_visitor::visit(shuffle_relation)");
  GQE_LOG_TRACE("Shuffle columns for shuffle relation: {}",
                logical::utility::list_to_string(relation->shuffle_cols_unsafe()));

  auto const children = relation->children_unsafe();
  assert(children.size() == 1);

  // Generate the children tasks and the partition tasks
  _generated_tasks =
    _builder->generate_partition_tasks(children[0], relation->shuffle_cols_unsafe());

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::project_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

  // Generate the project tasks
  for (auto& task : input_tasks) {
    std::vector<std::unique_ptr<expression>> exprs;
    for (auto const& expr : relation->output_expressions_unsafe())
      exprs.push_back(expr->clone());

    _generated_tasks.push_back(std::make_shared<project_task>(_builder->_ctx_ref,
                                                              _builder->_current_task_id,
                                                              _builder->_current_stage_id,
                                                              std::move(task),
                                                              std::move(exprs)));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::concatenate_sort_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto concatenated_input = _builder->concatenate(_builder->generate_tasks(children[0]));

  // Generate the sort task
  std::vector<std::unique_ptr<expression>> keys;
  keys.reserve(relation->keys_unsafe().size());
  for (auto const& key : relation->keys_unsafe())
    keys.push_back(key->clone());

  _generated_tasks.push_back(std::make_shared<sort_task>(_builder->_ctx_ref,
                                                         _builder->_current_task_id,
                                                         _builder->_current_stage_id,
                                                         std::move(concatenated_input),
                                                         std::move(keys),
                                                         relation->column_orders(),
                                                         relation->null_precedences()));
  _builder->_current_task_id++;

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::filter_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

  // Generate the filter tasks
  for (auto& input_task : input_tasks) {
    _generated_tasks.push_back(std::make_shared<filter_task>(_builder->_ctx_ref,
                                                             _builder->_current_task_id,
                                                             _builder->_current_stage_id,
                                                             std::move(input_task),
                                                             relation->condition_unsafe()->clone(),
                                                             relation->projection_indices()));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::window_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto concatenated_input = _builder->concatenate(_builder->generate_tasks(children[0]));

  cudf::aggregation::Kind aggr_func = relation->aggr_func();
  std::vector<std::unique_ptr<expression>> ident_cols;
  std::vector<std::unique_ptr<expression>> arguments;
  std::vector<std::unique_ptr<expression>> partition_by;
  std::vector<std::unique_ptr<expression>> order_by;

  for (auto const& ident_col : relation->ident_cols_unsafe()) {
    ident_cols.push_back(ident_col->clone());
  }
  for (auto const& argument : relation->arguments_unsafe()) {
    arguments.push_back(argument->clone());
  }
  for (auto const& partition_by_exp : relation->partition_by_unsafe()) {
    partition_by.push_back(partition_by_exp->clone());
  }
  for (auto const& order_by_exp : relation->order_by_unsafe()) {
    order_by.push_back(order_by_exp->clone());
  }
  auto order_dirs         = relation->order_dirs();
  auto window_lower_bound = relation->window_lower_bound();
  auto window_upper_bound = relation->window_upper_bound();

  _generated_tasks.push_back(std::make_shared<window_task>(_builder->_ctx_ref,
                                                           _builder->_current_task_id,
                                                           _builder->_current_stage_id,
                                                           std::move(concatenated_input),
                                                           aggr_func,
                                                           std::move(ident_cols),
                                                           std::move(arguments),
                                                           std::move(partition_by),
                                                           std::move(order_by),
                                                           std::move(order_dirs),
                                                           window_lower_bound,
                                                           window_upper_bound));
  _builder->_current_task_id++;

  update_cache(relation);
}

using aggregation_keys_type = std::vector<std::unique_ptr<expression>>;
using aggregation_values_type =
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>>;

namespace {

aggregation_keys_type clone_aggregation_keys(aggregation_keys_type const& other_keys)
{
  aggregation_keys_type aggregation_keys;
  aggregation_keys.reserve(other_keys.size());
  for (auto const& key : other_keys)
    aggregation_keys.push_back(key->clone());
  return aggregation_keys;
}

aggregation_values_type clone_aggregation_values(aggregation_values_type const& other_values)
{
  aggregation_values_type aggregation_values;
  aggregation_values.reserve(other_values.size());
  for (auto const& [kind, expr] : other_values)
    aggregation_values.emplace_back(kind, expr->clone());
  return aggregation_values;
}

}  // namespace

// Mappings for apply-concat-apply from the kind of the first aggregation to the kind of the
// second aggregation
cudf::aggregation::Kind get_second_aggregation_kind(cudf::aggregation::Kind first_aggregation_kind)
{
  switch (first_aggregation_kind) {
    case cudf::aggregation::SUM: return cudf::aggregation::SUM;
    case cudf::aggregation::COUNT_VALID: return cudf::aggregation::SUM;
    case cudf::aggregation::COUNT_ALL: return cudf::aggregation::SUM;
    case cudf::aggregation::MIN: return cudf::aggregation::MIN;
    case cudf::aggregation::MAX: return cudf::aggregation::MAX;
    default:
      throw std::logic_error("Unknown aggregation type in get_second_aggregation_kind: " +
                             std::to_string(first_aggregation_kind));
  }
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::concatenate_aggregate_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

  // FIXME: Apply-concat-apply is not necessary when input_tasks.size() == 1

  // Implement apply-concat-apply
  // There are mainly 4 steps:
  // Step 1: Apply aggregations on each partition
  // Step 2: Concatenate all partitions into a single partition
  // Step 3: Apply aggregations on the concatenated partition
  // Step 4: Optional post processing
  // For example, for "mean" aggregation, step 1 will perform "sum" and "count", step 3 will
  // perform "sum", and postprocessing will divide the sum by count.
  // Another example, for "count" aggregation, step 1 will perform "count", step 3 will perform
  // "sum", and there is no postprocessing.

  auto const keys   = relation->keys_unsafe();
  auto const values = relation->values_unsafe();

  // Step 1: Apply aggregations on each partition
  // Keys for the first aggregation are the same as those in the aggregation relation
  aggregation_keys_type first_aggregation_keys;
  first_aggregation_keys.reserve(keys.size());
  for (auto const& key : keys)
    first_aggregation_keys.push_back(key->clone());

  // Values for the first aggregation depend on the kind of the aggregation relation
  aggregation_values_type first_aggregation_values;
  first_aggregation_values.reserve(values.size());

  for (auto const& [kind, expr] : values) {
    switch (kind) {
      case cudf::aggregation::SUM:
        first_aggregation_values.emplace_back(cudf::aggregation::SUM, expr->clone());
        break;
      case cudf::aggregation::MEAN:
        first_aggregation_values.emplace_back(cudf::aggregation::SUM, expr->clone());
        first_aggregation_values.emplace_back(cudf::aggregation::COUNT_VALID, expr->clone());
        break;
      case cudf::aggregation::COUNT_VALID:
        first_aggregation_values.emplace_back(cudf::aggregation::COUNT_VALID, expr->clone());
        break;
      case cudf::aggregation::COUNT_ALL:
        first_aggregation_values.emplace_back(cudf::aggregation::COUNT_ALL, expr->clone());
        break;
      case cudf::aggregation::MIN:
        first_aggregation_values.emplace_back(cudf::aggregation::MIN, expr->clone());
        break;
      case cudf::aggregation::MAX:
        first_aggregation_values.emplace_back(cudf::aggregation::MAX, expr->clone());
        break;
      default:
        throw std::logic_error("Unknown aggregation type in task_graph_builder: " +
                               std::to_string(kind));
    }
  }

  std::vector<std::shared_ptr<task>> first_aggregation_tasks;
  first_aggregation_tasks.reserve(input_tasks.size());

  bool use_perfect_hashing =
    relation->perfect_hashing() &&
    _builder->_ctx_ref._query_context->parameters.aggregation_use_perfect_hash;

  for (auto& input_task : input_tasks) {
    auto condition = relation->condition_unsafe() ? relation->condition_unsafe()->clone()
                                                  : std::unique_ptr<expression>();
    first_aggregation_tasks.push_back(
      std::make_shared<aggregate_task>(_builder->_ctx_ref,
                                       _builder->_current_task_id,
                                       _builder->_current_stage_id,
                                       std::move(input_task),
                                       clone_aggregation_keys(first_aggregation_keys),
                                       clone_aggregation_values(first_aggregation_values),
                                       std::move(condition),
                                       use_perfect_hashing));
    _builder->_current_task_id++;
  }

  // Steps 3 and 4 for one input: the concatenated partial results, or in spill mode one partition
  // of them.
  auto const generate_final_aggregation = [&](std::shared_ptr<task> partial_results,
                                              bool final_perfect_hashing) -> std::shared_ptr<task> {
    // Step 3: Apply aggregations on the concatenated partition
    // Keys for the second aggregation are the first few columns of the concatenated table
    aggregation_keys_type second_aggregation_keys;
    second_aggregation_keys.reserve(keys.size());
    for (std::size_t column_idx = 0; column_idx < keys.size(); column_idx++)
      second_aggregation_keys.push_back(std::make_unique<column_reference_expression>(column_idx));

    // Values for the second aggregation depend on the kind of the first aggregation
    aggregation_values_type second_aggregation_values;
    second_aggregation_values.reserve(first_aggregation_values.size());
    for (std::size_t column_idx = 0; column_idx < first_aggregation_values.size(); column_idx++) {
      second_aggregation_values.emplace_back(
        get_second_aggregation_kind(first_aggregation_values[column_idx].first),
        std::make_unique<column_reference_expression>(keys.size() + column_idx));
    }

    auto second_aggregation_task =
      std::make_shared<aggregate_task>(_builder->_ctx_ref,
                                       _builder->_current_task_id,
                                       _builder->_current_stage_id,
                                       std::move(partial_results),
                                       std::move(second_aggregation_keys),
                                       std::move(second_aggregation_values),
                                       nullptr,
                                       final_perfect_hashing);
    _builder->_current_task_id++;

    // Step 4: Post processing
    std::vector<std::unique_ptr<expression>> output_expressions;

    // Keys don't need post-processing
    for (std::size_t column_idx = 0; column_idx < keys.size(); column_idx++)
      output_expressions.push_back(std::make_unique<column_reference_expression>(column_idx));

    std::size_t in_idx = keys.size();
    for (auto const& value : values) {
      switch (value.first) {
        case cudf::aggregation::SUM:
        case cudf::aggregation::COUNT_VALID:
        case cudf::aggregation::COUNT_ALL:
        case cudf::aggregation::MIN:
        case cudf::aggregation::MAX:
          // SUM, COUNT_VALID, COUNT_ALL, MIN, MAX do not need post-processing
          output_expressions.push_back(std::make_unique<column_reference_expression>(in_idx));
          in_idx++;
          break;
        case cudf::aggregation::MEAN:
          // MEAN needs to divide the SUM by the COUNT_VALID
          output_expressions.push_back(std::make_unique<divide_expression>(
            std::make_shared<column_reference_expression>(in_idx),
            std::make_shared<column_reference_expression>(in_idx + 1)));
          in_idx += 2;
          break;
        default:
          throw std::logic_error("Unknown aggregation type in task_graph_builder: " +
                                 std::to_string(value.first));
      }
    }
    assert(in_idx == keys.size() + first_aggregation_values.size());

    auto post_processing_task = std::make_shared<project_task>(_builder->_ctx_ref,
                                                               _builder->_current_task_id,
                                                               _builder->_current_stage_id,
                                                               std::move(second_aggregation_task),
                                                               std::move(output_expressions));
    _builder->_current_task_id++;
    return post_processing_task;
  };

  // flowcsd spill mode: keyed partial results go through the hybrid path (executor/spill.hpp):
  // partitioned, resident while they fit the budget, the rest on SSD, and one final aggregation per
  // partition instead of one over the concatenation of all partial results.
  bool const spill_partials = get_spill_settings().partitions > 0 && !keys.empty();
  if (spill_partials) {
    std::vector<std::string> input_tables;
    auto const input_files =
      count_input_files(children[0], _builder->_table_provider, input_tables);
    auto const description = std::format("aggregation keys={} input_tables={} input_files={}",
                                         keys.size(),
                                         join_names(input_tables, ","),
                                         input_files);
    auto const signature =
      _builder->next_spill_signature("aggregation", std::move(input_tables), keys.size());
    auto const operator_partition_count = spill_partitions_for(signature);
    auto const partitions               = operator_partition_count.partitions;
    GQE_LOG_INFO("flowcsd_spill_plan {} hybrid=true signature={} partitions={} partitions_from={}",
                 description,
                 signature,
                 partitions,
                 operator_partition_count.from_table ? "table" : "default");
    count_hybrid_operator(hybrid_operator::aggregation);
    auto const& parameters = _builder->_ctx_ref._query_context->parameters;
    // The resident slices are copied on the compute thread that partitions them, unsynchronized for
    // other workers' streams.
    GQE_EXPECTS(parameters.num_workers == 1, "spill mode needs GQE_NUM_WORKERS=1");
    // The partial results start with their key columns (step 3).
    std::vector<std::unique_ptr<expression>> partial_keys;
    for (std::size_t column_idx = 0; column_idx < keys.size(); column_idx++)
      partial_keys.push_back(std::make_unique<column_reference_expression>(column_idx));
    auto const pool_bytes =
      parameters.max_query_memory.value_or(rmm::available_device_memory().second);
    spill_operator_plan plan;
    plan.num_sides             = 1;
    plan.description           = description;
    plan.signature             = signature;
    plan.partitions            = partitions;
    plan.held_side             = 0;
    plan.held_input_tasks      = static_cast<int64_t>(first_aggregation_tasks.size());
    plan.resident_budget_bytes = spill_resident_budget_bytes(pool_bytes);
    plan.pool_bytes            = pool_bytes;
    auto spill                 = make_spill_set(std::move(plan));
    auto partition_tasks       = _builder->generate_spill_partition_tasks(
      std::move(first_aggregation_tasks), utility::to_const_raw_ptrs(partial_keys), 0, spill, 0);
    // Stage C: one final aggregation per partition; partition 0 brings every resident partition,
    // and partitions without spill data do not run (spill_partition_done_task).
    auto const output_types = relation->output_data_types();
    for (int32_t partition = 0; partition < partitions; partition++) {
      auto partial_results = std::make_shared<spill_read_task>(_builder->_ctx_ref,
                                                               _builder->_current_task_id,
                                                               _builder->_current_stage_id,
                                                               partition_tasks,
                                                               spill,
                                                               /*side=*/0,
                                                               partition);
      _builder->_current_task_id++;
      // Perfect hashing is off for the same reason as in generate_spilled_join.
      auto final_aggregation = generate_final_aggregation(std::move(partial_results), false);
      _generated_tasks.push_back(std::make_shared<spill_partition_done_task>(_builder->_ctx_ref,
                                                                             _builder->_current_task_id,
                                                                             _builder->_current_stage_id,
                                                                             std::move(final_aggregation),
                                                                             spill,
                                                                             partition,
                                                                             spill_match_rule::unknown,
                                                                             output_types));
      _builder->_current_task_id++;
    }
  } else {
    // Step 2: Concatenate all partitions into a single partition
    auto concatenated_task = _builder->concatenate(std::move(first_aggregation_tasks));
    _generated_tasks.push_back(
      generate_final_aggregation(std::move(concatenated_task), use_perfect_hashing));
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::fetch_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto concatenated_input = _builder->concatenate(_builder->generate_tasks(children[0]));

  if (relation->offset() > std::numeric_limits<cudf::size_type>::max() ||
      relation->count() > std::numeric_limits<cudf::size_type>::max())
    throw std::overflow_error(
      "Offset or limit overflows when building a task graph for a fetch relation");

  _generated_tasks.push_back(std::make_shared<fetch_task>(_builder->_ctx_ref,
                                                          _builder->_current_task_id,
                                                          _builder->_current_stage_id,
                                                          std::move(concatenated_input),
                                                          relation->offset(),
                                                          relation->count()));
  _builder->_current_task_id++;

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::gen_ident_col_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 1);
  auto input_tasks = _builder->generate_tasks(children[0]);

  for (auto& task : input_tasks) {
    _generated_tasks.push_back(std::make_shared<gen_ident_col_task>(_builder->_ctx_ref,
                                                                    _builder->_current_task_id,
                                                                    _builder->_current_stage_id,
                                                                    std::move(task)));
    _builder->_current_task_id++;
  }

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(physical::union_all_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the input tasks
  auto const children = relation->children_unsafe();
  assert(children.size() == 2);
  auto left_tasks = _builder->generate_tasks(children[0]);
  _builder->insert_pipeline_breaker(left_tasks);
  auto right_tasks = _builder->generate_tasks(children[1]);

  // Concatenate the left tasks and right tasks
  for (auto& task : left_tasks)
    _generated_tasks.push_back(std::move(task));

  for (auto& task : right_tasks)
    _generated_tasks.push_back(std::move(task));

  update_cache(relation);
}

void task_graph_builder::generate_task_graph_visitor::visit(
  physical::user_defined_relation* relation)
{
  if (is_cached(relation)) return;

  // Recursively generate the tasks for child relations
  auto const children                  = relation->children_unsafe();
  auto const last_child_break_pipeline = relation->last_child_break_pipeline();

  std::vector<std::vector<std::shared_ptr<task>>> children_tasks;
  for (std::size_t child_idx = 0; child_idx < children.size(); child_idx++) {
    auto child_task = _builder->generate_tasks(children[child_idx]);
    if (child_idx != children.size() - 1 || last_child_break_pipeline)
      _builder->insert_pipeline_breaker(child_task);
    children_tasks.push_back(std::move(child_task));
  }

  // Generate tasks for the user-defined relation through the user specified functor
  int32_t task_id  = _builder->_current_task_id;
  _generated_tasks = relation->task_functor()(
    std::move(children_tasks), _builder->_ctx_ref, task_id, _builder->_current_stage_id);
  _builder->_current_task_id = task_id;

  update_cache(relation);
}

bool task_graph_builder::generate_task_graph_visitor::is_cached(physical::relation* relation)
{
  if (_generated_tasks.size() != 0)
    throw std::logic_error("generate_task_graph_visitor already contains generated tasks");

  // Check the cache to see whether tasks for `relation` have already been generated
  auto tasks_cache_iter = _builder->_tasks_cache.find(relation);
  if (tasks_cache_iter != _builder->_tasks_cache.end()) {
    for (auto const& task_ptr : tasks_cache_iter->second)
      _generated_tasks.emplace_back(task_ptr);
    return true;
  }

  return false;
}

void task_graph_builder::generate_task_graph_visitor::update_cache(physical::relation* relation)
{
  std::vector<std::weak_ptr<task>> tasks;
  for (auto const& task_ptr : _generated_tasks)
    tasks.emplace_back(task_ptr);

  _builder->_tasks_cache[relation] = std::move(tasks);
}

}  // namespace gqe
