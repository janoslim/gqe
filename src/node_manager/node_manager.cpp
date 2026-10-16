/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/node_manager/node_manager.hpp>

#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/node_manager/log_drain.hpp>
#include <gqe/node_manager/log_stream_service.hpp>
#include <gqe/node_manager/nvshmem_bootstrap.hpp>
#include <gqe/node_manager/service.hpp>
#include <gqe/node_manager/spawn.hpp>
#include <gqe/node_manager/stats_tables.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/optimization_configuration.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/rpc/serialization/optimization_parameters.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>
#include <gqe/rpc/serialization/statistics.hpp>
#include <gqe/rpc/serialization/storage.hpp>
#include <gqe/rpc/serialization/uuid.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/uuid.hpp>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <arrow/flight/types.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <signal.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <future>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace gqe::node_manager {

namespace {

/**
 * Visitor that collects table names from read and write relations in a physical plan.
 * The base visitor's default implementations recurse into children and subqueries.
 */
struct table_name_collector : public physical::relation_visitor {
  std::unordered_set<std::string> table_names;

  void visit(physical::read_relation* rel) override
  {
    table_names.insert(rel->table_name());
    visit_children(rel);
  }
  void visit(physical::write_relation* rel) override
  {
    table_names.insert(rel->table_name());
    visit_children(rel);
  }
};

/// Serialize a record batch to an Arrow IPC stream buffer.
std::string serialize_record_batch_to_ipc(arrow::RecordBatch const& batch)
{
  auto sink   = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer = arrow::ipc::MakeStreamWriter(sink, batch.schema()).ValueOrDie();
  auto status = writer->WriteRecordBatch(batch);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to write stats Arrow IPC: {}", status.ToString()));
  }
  status = writer->Close();
  if (!status.ok()) {
    throw std::runtime_error(
      std::format("Failed to close stats Arrow IPC writer: {}", status.ToString()));
  }
  auto buffer = sink->Finish().ValueOrDie();
  return std::string(reinterpret_cast<char const*>(buffer->data()), buffer->size());
}

/**
 * @brief Return a copy of `batch` with its columns renamed to `names`.
 *
 * Column types and data are preserved; only the schema field names change. `names` are the
 * query's real output column names (the Substrait RelRoot.names captured during parsing).
 *
 * The rename is guarded:
 * - If `names` is empty or its size doesn't match the column count, `batch` is returned
 *   unchanged, keeping the placeholder names the executor emits (column_0, column_1, ...).
 * - Count and names always match for the flat output schemas GQE supports; nested structs, which
 *   could make the flattened name count differ from the column count, are rejected upstream.
 *
 * @param batch The result batch whose columns to relabel.
 * @param names The output column names to apply, one per column.
 * @return A copy of `batch` with `names` applied, or `batch` unchanged if the guard trips.
 */
std::shared_ptr<arrow::RecordBatch> relabel_batch_columns(std::shared_ptr<arrow::RecordBatch> batch,
                                                          std::vector<std::string> const& names)
{
  if (!batch || names.size() != static_cast<std::size_t>(batch->num_columns())) { return batch; }

  arrow::FieldVector fields;
  fields.reserve(names.size());
  for (int i = 0; i < batch->num_columns(); ++i) {
    fields.push_back(batch->schema()->field(i)->WithName(names[i]));
  }
  return arrow::RecordBatch::Make(
    arrow::schema(std::move(fields)), batch->num_rows(), batch->columns());
}

}  // namespace

// ============================================================================
// Construction / destruction
// ============================================================================

node_manager::node_manager(configuration config, std::shared_ptr<log_drain> drain)
  : _config(std::move(config)), _log_drain(std::move(drain))
{
  if (!_log_drain) { throw std::invalid_argument("Node manager log drain cannot be null"); }
  if (_config.task_manager_base_port == 0) { _config.task_manager_base_port = _config.port + 1000; }

  _ctx     = std::make_unique<context>(_config.opt_params);
  _catalog = std::make_unique<gqe::catalog>(_ctx.get());

  // Always-present system tables that expose statistics to SQL clients; populated lazily on read.
  register_stats_tables(*_catalog);
}

node_manager::~node_manager() { shutdown(); }

// ============================================================================
// Serve / shutdown
// ============================================================================

arrow::Status node_manager::serve()
{
  _log_stream_service = std::make_unique<log_stream_service>(_log_drain);
  grpc::ServerBuilder log_server_builder;
  apply_default_server_args(log_server_builder);
  int log_port = 0;
  log_server_builder.AddListeningPort(
    std::format("{}:0", grpc_localhost_addr), grpc::InsecureServerCredentials(), &log_port);
  log_server_builder.RegisterService(_log_stream_service.get());
  _log_server = log_server_builder.BuildAndStart();
  if (!_log_server || log_port == 0) {
    return arrow::Status::IOError("Failed to start node-manager log server");
  }
  _log_address = std::format("{}:{}", grpc_localhost_addr, log_port);
  GQE_LOG_INFO("Log server listening on {}", _log_address);

  {
    std::scoped_lock guard(_task_manager_mutex);
    launch_task_managers_unsafe();
  }
  _monitor_thread =
    std::jthread([this](std::stop_token token) { monitor_task_managers(std::move(token)); });

  // Create the Flight service with execution callbacks.
  _service = std::make_unique<service>(
    _catalog.get(),
    [this](void const* data, std::size_t size) { return execute_substrait_query(data, size); },
    [this](void const* data, std::size_t size) { return execute_substrait_statement(data, size); },
    [this](void const* data, std::size_t size) { return execute_physical_plan_query(data, size); },
    [this](std::string const& name, arrow::flight::SessionOptionValue const& value) {
      std::scoped_lock guard(_execute_mutex);
      return rpc::apply_session_option(_config.opt_params, name, value);
    },
    [this]() -> arrow::Result<std::map<std::string, arrow::flight::SessionOptionValue>> {
      std::scoped_lock guard(_execute_mutex);
      return rpc::optimization_parameters_to_session_options(_config.opt_params);
    });

  // Start Flight SQL server
  auto location = arrow::flight::Location::ForGrpcTcp(_config.listen_address, _config.port);
  if (!location.ok()) { return location.status(); }

  arrow::flight::FlightServerOptions options(*location);

  GQE_LOG_INFO("Node manager starting on {}:{} with {} GPU(s)",
               _config.listen_address,
               _config.port,
               _config.num_gpus);

  ARROW_RETURN_NOT_OK(_service->Init(options));
  ARROW_RETURN_NOT_OK(_service->SetShutdownOnSignals({SIGINT, SIGTERM, SIGHUP}));
  ARROW_RETURN_NOT_OK(_service->Serve());

  // Serve() returned — either a signal was received or Shutdown() was called.
  shutdown();
  return arrow::Status::OK();
}

void node_manager::shutdown()
{
  // The log server starts before the task-manager monitor. Keep shutdown idempotence independent
  // of the monitor's stop state so an early launch or bootstrap failure still releases the server.
  if (_shutdown_started.exchange(true, std::memory_order_acq_rel)) { return; }

  if (_monitor_thread.joinable()) {
    _monitor_thread.request_stop();
    _monitor_cv.notify_all();
    _monitor_thread.join();
  }

  constexpr auto task_manager_shutdown_timeout = std::chrono::seconds(5);

  GQE_LOG_CRITICAL("Node manager shutting down");

  // Stop the Flight SQL server so Serve() returns.
  if (_service) {
    auto status = _service->Shutdown();
    if (!status.ok()) { GQE_LOG_WARN("Flight SQL server shutdown failed: {}", status.ToString()); }
  }

  cancel_in_flight_rpcs();

  {
    std::scoped_lock guard(_task_manager_mutex);

    // Close gRPC channels so task managers can drain quickly.
    _task_manager_channels.clear();

    GQE_LOG_INFO("Terminating task managers");
    _task_manager_processes.terminate_all(task_manager_shutdown_timeout);
    GQE_LOG_INFO("All task managers terminated");
  }

  if (_log_server) {
    _log_server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds{1});
    _log_server->Wait();
    _log_server.reset();
  }
}

// ============================================================================
// In-flight RPC cancellation
// ============================================================================

void node_manager::cancel_in_flight_rpcs()
{
  std::scoped_lock guard(_in_flight_mutex);
  for (auto* ctx : _in_flight_rpcs) {
    ctx->TryCancel();
  }
}

// ============================================================================
// Task manager subprocess management
// ============================================================================

void node_manager::launch_task_managers_unsafe()
{
  // Assert empty rather than silently terminating existing child processes
  // via move-assignment on process_group.
  GQE_EXPECTS(_task_manager_processes.size() == 0,
              "Cannot launch task managers while existing processes are still active");
  auto spawned = spawn_task_managers(
    _config.task_manager_binary, _config.task_manager_base_port, _config.num_gpus, _log_address);
  _task_manager_processes = std::move(spawned.processes);
  _task_manager_channels  = std::move(spawned.channels);

  bootstrap_nvshmem(_task_manager_channels);
}

void node_manager::restart_task_managers()
{
  GQE_LOG_WARN("Restarting all task managers");

  _task_manager_processes.terminate_all();

  launch_task_managers_unsafe();
}

void node_manager::monitor_task_managers(std::stop_token token)
{
  while (!token.stop_requested()) {
    {
      std::scoped_lock guard(_task_manager_mutex);
      auto status = _task_manager_processes.try_wait_any();
      if (status) {
        switch (status->kind) {
          case exit_kind::exited:
            GQE_LOG_ERROR("Task manager rank {} (pid {}) exited with status {}",
                          status->rank,
                          status->pid,
                          status->code);
            break;
          case exit_kind::signaled:
            GQE_LOG_ERROR("Task manager rank {} (pid {}) killed by signal {}",
                          status->rank,
                          status->pid,
                          status->code);
            break;
        }

        cancel_in_flight_rpcs();
        restart_task_managers();
      }
    }
    std::unique_lock lock(_monitor_mutex);
    _monitor_cv.wait_for(lock, token, std::chrono::seconds(1), [] { return false; });
  }
}

// ============================================================================
// Multi-GPU query execution
// ============================================================================

node_manager::execution_result node_manager::execute_on_task_managers(
  std::shared_ptr<gqe::physical::relation> physical_plan)
{
  auto serialized_plan       = rpc::serialize_physical_plan(physical_plan.get());
  auto serialized_opt_params = rpc::serialize_optimization_parameters(_config.opt_params);
  auto query_id              = utility::uuid::generate();

  // Build the table catalog: only the tables referenced by this plan.
  table_name_collector collector;
  physical_plan->accept(collector);
  auto const& referenced_tables = collector.table_names;
  google::protobuf::RepeatedPtrField<proto::StorageDescriptor> table_catalog;
  for (auto const& name : referenced_tables) {
    *table_catalog.Add() = rpc::serialize_storage_descriptor(_catalog->storage_descriptor(name));
  }

  // Snapshot channels under the lock so that a concurrent restart does not
  // mutate the vector while we are iterating over it.
  std::vector<std::shared_ptr<grpc::Channel>> channels;
  {
    std::scoped_lock guard(_task_manager_mutex);
    channels = _task_manager_channels;
  }

  struct in_flight_rpc {
    grpc::ClientContext ctx;
    std::future<proto::ExecutePlanResponse> future;
  };
  std::vector<in_flight_rpc> rpcs(channels.size());

  for (std::size_t i = 0; i < channels.size(); ++i) {
    rpcs[i].ctx.set_deadline(std::chrono::system_clock::now() + _config.query_timeout);

    {
      std::scoped_lock guard(_in_flight_mutex);
      _in_flight_rpcs.push_back(&rpcs[i].ctx);
    }

    rpcs[i].future = std::async(std::launch::async, [&, i]() {
      proto::ExecutePlanRequest request;
      *request.mutable_physical_plan()           = serialized_plan;
      *request.mutable_query_id()                = rpc::serialize_uuid(query_id);
      *request.mutable_table_catalog()           = table_catalog;
      *request.mutable_optimization_parameters() = serialized_opt_params;

      GQE_LOG_TRACE("Node manager: submitting query {} to rank {}", query_id, i);
      proto::ExecutePlanResponse response;
      auto stub   = proto::TaskManagerService::NewStub(channels[i]);
      auto status = stub->ExecutePlan(&rpcs[i].ctx, request, &response);
      GQE_LOG_TRACE("Node manager: received response for query {} from rank {} (success={})",
                    query_id,
                    i,
                    status.ok());
      if (!status.ok()) {
        response.set_success(false);
        response.set_error_message(status.error_message());
      }
      return response;
    });
  }

  // Collect results from all ranks
  std::vector<std::shared_ptr<arrow::RecordBatch>> partial_results;
  gqe::table_statistics aggregated_write_stats;
  bool has_write_stats = false;

  // See the task manager error contract in the execute_on_task_managers() Doxygen.
  std::string first_error;
  std::string fatal_error;

  for (std::size_t i = 0; i < rpcs.size(); ++i) {
    auto response = rpcs[i].future.get();
    if (!response.success()) {
      auto message = std::format("Task manager rank {} failed: {}", i, response.error_message());
      if (response.fatal()) {
        fatal_error = std::move(message);
        for (std::size_t j = i + 1; j < rpcs.size(); ++j) {
          rpcs[j].ctx.TryCancel();
        }
        break;
      }
      if (first_error.empty()) { first_error = std::move(message); }
      continue;
    }
    if (response.has_write_statistics()) {
      has_write_stats = true;
      aggregated_write_stats.append_table_statistics(
        rpc::deserialize_table_statistics(response.write_statistics()));
    }
    if (!response.arrow_ipc_result().empty()) {
      // Move IPC bytes out of the response — response.arrow_ipc_result() is invalid after this.
      auto owned_buffer =
        arrow::Buffer::FromString(std::move(*response.mutable_arrow_ipc_result()));
      auto reader = arrow::ipc::RecordBatchStreamReader::Open(
                      std::make_shared<arrow::io::BufferReader>(owned_buffer))
                      .ValueOrDie();
      auto batch = reader->Next().ValueOrDie();
      if (batch) { partial_results.push_back(batch); }
    }
  }

  // Unregister in-flight RPCs
  {
    std::scoped_lock guard(_in_flight_mutex);
    for (auto& rpc : rpcs) {
      std::erase(_in_flight_rpcs, &rpc.ctx);
    }
  }

  if (!fatal_error.empty()) {
    {
      std::scoped_lock guard(_task_manager_mutex);
      restart_task_managers();
    }
    throw std::runtime_error(fatal_error);
  }

  if (!first_error.empty()) { throw std::runtime_error(first_error); }

  auto write_stats = has_write_stats ? std::optional{aggregated_write_stats} : std::nullopt;

  if (partial_results.empty()) { return {nullptr, write_stats}; }

  auto table    = arrow::Table::FromRecordBatches(partial_results).ValueOrDie();
  auto combined = table->CombineChunks().ValueOrDie();
  auto reader   = std::make_shared<arrow::TableBatchReader>(*combined);
  std::shared_ptr<arrow::RecordBatch> batch;
  auto st = reader->ReadNext(&batch);
  if (!st.ok()) {
    throw std::runtime_error(std::format("Failed to read combined batch: {}", st.ToString()));
  }
  return {batch, write_stats};
}

void node_manager::refresh_stats_tables_if_stale(
  std::unordered_set<std::string> const& referenced_tables)
{
  bool touches_stats = false;
  for (auto const& name : referenced_tables) {
    if (is_stats_table(name)) {
      touches_stats = true;
      break;
    }
  }
  if (!touches_stats) { return; }

  // The catalog statistics version only advances on a statistics change (write) or a table being
  // registered/unregistered. If nothing changed since the last materialization, the shared tables
  // are already current.
  auto const current_version = _catalog->stats_version();
  if (_materialized_stats_version == current_version) { return; }

  GQE_LOG_TRACE("Refreshing statistics tables (version {} -> {})",
                _materialized_stats_version.value_or(0),
                current_version);

  // Project the current statistics into Arrow batches and build the refresh request once; it is
  // identical for every rank (rank 0 writes the shared copy, others update local references).
  proto::RefreshStatsRequest request;
  *request.mutable_query_id() = rpc::serialize_uuid(utility::uuid::generate());
  request.set_epoch(current_version);

  auto const table_stats_name  = std::string{table_stats_table_name};
  auto const column_stats_name = std::string{column_stats_table_name};

  auto* table_write = request.add_writes();
  table_write->set_table_name(table_stats_name);
  table_write->set_arrow_ipc(serialize_record_batch_to_ipc(*project_table_stats(*_catalog)));

  auto* column_write = request.add_writes();
  column_write->set_table_name(column_stats_name);
  column_write->set_arrow_ipc(serialize_record_batch_to_ipc(*project_column_stats(*_catalog)));

  *request.add_table_catalog() =
    rpc::serialize_storage_descriptor(_catalog->storage_descriptor(table_stats_name));
  *request.add_table_catalog() =
    rpc::serialize_storage_descriptor(_catalog->storage_descriptor(column_stats_name));

  // Snapshot channels under the lock so a concurrent restart does not mutate the vector.
  std::vector<std::shared_ptr<grpc::Channel>> channels;
  {
    std::scoped_lock guard(_task_manager_mutex);
    channels = _task_manager_channels;
  }
  if (channels.empty()) {
    _materialized_stats_version = current_version;
    return;
  }

  // Only rank 0 publishes the shared statistics row groups; every other rank reattaches to the
  // shared contents on its next read of the table (that read's descriptor carries the bumped
  // storage_revision — see remote_table_cache::update). So the data-carrying refresh only needs to
  // reach rank 0. Channel index matches task-manager rank throughout the node manager (see
  // execute_on_task_managers), so channels[0] is rank 0; the task-manager side additionally guards
  // the write on nvshmem_rank_zero().
  struct in_flight_rpc {
    grpc::ClientContext ctx;
    std::future<proto::RefreshStatsResponse> future;
  };
  in_flight_rpc rpc;
  rpc.ctx.set_deadline(std::chrono::system_clock::now() + _config.query_timeout);
  {
    std::scoped_lock guard(_in_flight_mutex);
    _in_flight_rpcs.push_back(&rpc.ctx);
  }
  rpc.future = std::async(std::launch::async, [&]() {
    proto::RefreshStatsResponse response;
    auto stub   = proto::TaskManagerService::NewStub(channels[0]);
    auto status = stub->RefreshStatsTables(&rpc.ctx, request, &response);
    if (!status.ok()) {
      response.set_success(false);
      response.set_error_message(status.error_message());
    }
    return response;
  });

  // Barrier on rank 0's write before the caller's dependent read.
  auto response = rpc.future.get();
  {
    std::scoped_lock guard(_in_flight_mutex);
    std::erase(_in_flight_rpcs, &rpc.ctx);
  }
  if (!response.success()) {
    throw std::runtime_error(
      std::format("Task manager rank 0 stats refresh failed: {}", response.error_message()));
  }

  _materialized_stats_version = current_version;
}

// ============================================================================
// Substrait execution — delegates to task managers
// ============================================================================

namespace {

gqe::optimizer::optimization_configuration make_default_optimizer_rules()
{
  return gqe::optimizer::optimization_configuration(
    {gqe::optimizer::logical_optimization_rule_type::constant_folding,
     gqe::optimizer::logical_optimization_rule_type::projection_pushdown,
     gqe::optimizer::logical_optimization_rule_type::normalize_decimal_widths,
     gqe::optimizer::logical_optimization_rule_type::identity_project_elimination,
     gqe::optimizer::logical_optimization_rule_type::complex_expression_extraction_into_project,
     gqe::optimizer::logical_optimization_rule_type::join_predicate_extraction_into_project,
     gqe::optimizer::logical_optimization_rule_type::mean_decomposition,
     gqe::optimizer::logical_optimization_rule_type::string_to_int_literal,
     gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
     gqe::optimizer::logical_optimization_rule_type::join_unique_keys,
     gqe::optimizer::logical_optimization_rule_type::aggregate_perfect_hash,
     gqe::optimizer::logical_optimization_rule_type::join_perfect_hash,
     gqe::optimizer::logical_optimization_rule_type::fix_partial_filter_column_references},
    {});
}

// flowcsd observation: CLOCK_MONOTONIC nanoseconds (libstdc++ steady_clock), so query timestamps
// can be aligned with host-side samplers and traces that use the same clock.
int64_t monotonic_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

// flowcsd observation switch: FLOWCSD_TIMING=0 turns the timing record off (unset or any other
// value keeps it on), so the study can compare wall time with and without it on one build.
bool timing_enabled()
{
  static bool const enabled = [] {
    char const* value = std::getenv("FLOWCSD_TIMING");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

}  // namespace

std::shared_ptr<arrow::RecordBatch> node_manager::execute_substrait_query(void const* data,
                                                                          std::size_t size)
{
  bool const record_timing  = timing_enabled();
  auto const query_start_ns = record_timing ? monotonic_ns() : 0;
  std::scoped_lock guard(_execute_mutex);

  gqe::substrait_parser parser(_catalog.get(), &_config.opt_params);
  auto queries = parser.from_binary(data, size);

  if (queries.empty()) { return nullptr; }

  GQE_EXPECTS(queries.size() == 1,
              std::format("Expected exactly one parsed query relation, got {}", queries.size()));
  auto const& query = queries.front();

  if (query.relation->type() == gqe::logical::relation::relation_type::write) {
    throw std::logic_error("execute_substrait_query called with a write plan");
  }

  auto rule_config = make_default_optimizer_rules();
  gqe::optimizer::logical_optimizer optimizer(&rule_config, _catalog.get());

  GQE_LOG_TRACE("Catalog statistics before optimization:\n{}", _catalog->to_string());

  auto optimized = optimizer.optimize(query.relation);
  GQE_LOG_TRACE("Optimized logical plan:\n{}", optimized->to_string());

  gqe::physical_plan_builder plan_builder(_catalog.get(), &_config.opt_params);
  auto physical_plan     = plan_builder.build(optimized.get());
  auto const plan_end_ns = record_timing ? monotonic_ns() : 0;
  GQE_LOG_TRACE("Physical plan:\n{}", physical_plan->to_string());

  // If this query reads a statistics table, lazily re-materialize it from the catalog before the
  // read is dispatched (no-op if already current).
  table_name_collector collector;
  physical_plan->accept(collector);
  refresh_stats_tables_if_stale(collector.table_names);

  auto batch  = execute_on_task_managers(physical_plan).batch;
  auto result = relabel_batch_columns(std::move(batch), query.output_column_names);
  if (record_timing) {
    GQE_LOG_INFO("flowcsd_timing node_manager query_start_ns={} plan_end_ns={} query_end_ns={}",
                 query_start_ns,
                 plan_end_ns,
                 monotonic_ns());
  }
  return result;
}

std::optional<gqe::table_statistics> node_manager::execute_substrait_statement(void const* data,
                                                                               std::size_t size)
{
  std::scoped_lock guard(_execute_mutex);

  gqe::substrait_parser parser(_catalog.get(), &_config.opt_params);
  auto queries = parser.from_binary(data, size);

  if (queries.empty()) { return {}; }

  GQE_EXPECTS(queries.size() == 1,
              std::format("Expected exactly one parsed query relation, got {}", queries.size()));
  auto const& query = queries.front();

  if (query.relation->type() != gqe::logical::relation::relation_type::write) {
    throw std::logic_error("execute_substrait_statement called with a non-write plan");
  }

  auto rule_config = make_default_optimizer_rules();
  gqe::optimizer::logical_optimizer optimizer(&rule_config, _catalog.get());
  auto optimized = optimizer.optimize(query.relation);

  gqe::physical_plan_builder plan_builder(_catalog.get(), &_config.opt_params);
  auto physical_plan = plan_builder.build(optimized.get());

  auto result = execute_on_task_managers(physical_plan);

  // The catalog's shared-memory facade intentionally remains stale after a write. Publication
  // increments the storage revision, causing task managers to replace stale facades before the
  // next query.
  // TODO: Separate shared-storage ownership from process-local snapshots.

  if (result.write_stats.has_value()) {
    auto* write_rel = static_cast<gqe::physical::write_relation*>(physical_plan.get());
    _catalog->append_table_statistics(write_rel->table_name(), result.write_stats.value());
  }
  return result.write_stats;
}

std::shared_ptr<arrow::RecordBatch> node_manager::execute_physical_plan_query(void const* data,
                                                                              std::size_t size)
{
  std::lock_guard guard(_execute_mutex);

  proto::PhysicalRelation pr;
  if (!pr.ParseFromArray(data, static_cast<int>(size))) {
    throw std::runtime_error("Failed to parse PhysicalRelation from binary data");
  }

  auto physical_plan = gqe::rpc::deserialize_physical_plan(pr);
  if (physical_plan->type() == gqe::physical::relation::relation_type::write) {
    throw std::logic_error("execute_physical_plan_query called with a write plan");
  }

  // Re-materialize statistics tables on demand if this plan reads one (no-op otherwise).
  table_name_collector collector;
  physical_plan->accept(collector);
  refresh_stats_tables_if_stale(collector.table_names);

  return execute_on_task_managers(physical_plan).batch;
}

}  // namespace gqe::node_manager
