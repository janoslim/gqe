/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/task_manager/plan_executor.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/spill.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/query_context.hpp>
#include <gqe/rpc/serialization/statistics.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/cudf_to_arrow.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/multi_process_helpers.hpp>

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <string_view>

namespace gqe::task_manager {

namespace {

/// Serialize result views to Arrow IPC bytes.
std::string serialize_to_arrow_ipc(std::span<cudf::table_view const> result_views)
{
  std::vector<std::string> column_names;
  auto num_cols = result_views[0].num_columns();
  column_names.reserve(num_cols);
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    column_names.push_back(std::format("column_{}", i));
  }

  auto batch  = gqe::utility::cudf_table_to_arrow(result_views, column_names);
  auto sink   = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
  auto status = writer->WriteRecordBatch(*batch);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to write Arrow IPC: {}", status.ToString()));
  }
  status = writer->Close();
  if (!status.ok()) {
    throw std::runtime_error(
      std::format("Failed to close Arrow IPC writer: {}", status.ToString()));
  }
  auto buffer = sink->Finish().ValueOrDie();
  return std::string(reinterpret_cast<char const*>(buffer->data()), buffer->size());
}

/// Read the single record batch out of an Arrow IPC stream buffer.
std::shared_ptr<arrow::RecordBatch> deserialize_arrow_ipc(std::string const& bytes)
{
  auto buffer = arrow::Buffer::FromString(bytes);
  // Throw (rather than ValueOrDie's abort) on a malformed or incompatible payload, so the
  // RefreshStatsTables handler can report success=false instead of terminating the task manager.
  auto reader_result =
    arrow::ipc::RecordBatchStreamReader::Open(std::make_shared<arrow::io::BufferReader>(buffer));
  if (!reader_result.ok()) {
    throw std::runtime_error(
      std::format("Failed to open stats Arrow IPC stream: {}", reader_result.status().ToString()));
  }
  auto reader = std::move(reader_result).ValueOrDie();
  std::shared_ptr<arrow::RecordBatch> batch;
  auto status = reader->ReadNext(&batch);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to read stats Arrow IPC: {}", status.ToString()));
  }
  if (!batch) { throw std::runtime_error("Stats Arrow IPC stream contained no record batch"); }
  return batch;
}

/// Convert an Arrow record batch to a cuDF device table via the Arrow C data interface.
std::unique_ptr<cudf::table> arrow_batch_to_cudf(arrow::RecordBatch const& batch,
                                                 rmm::cuda_stream_view stream)
{
  ArrowSchema c_schema;
  ArrowArray c_array;
  auto status = arrow::ExportRecordBatch(batch, &c_array, &c_schema);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to export record batch: {}", status.ToString()));
  }
  auto table = cudf::from_arrow(&c_schema, &c_array, stream);
  // cudf::from_arrow copies the data; release the C structures we exported.
  if (c_array.release != nullptr) { c_array.release(&c_array); }
  if (c_schema.release != nullptr) { c_schema.release(&c_schema); }
  return table;
}

// flowcsd observation: CLOCK_MONOTONIC nanoseconds (libstdc++ steady_clock), so phase timestamps
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

plan_executor::plan_executor(task_manager_context* ctx, int rank, bool multi_process)
  : _ctx(ctx), _cache(ctx), _rank(rank), _multi_process(multi_process)
{
}

service::execution_result plan_executor::operator()(std::shared_ptr<physical::relation> plan,
                                                    utility::uuid query_id,
                                                    std::vector<storage::descriptor> descriptors,
                                                    gqe::optimization_parameters opt_params)
{
  GQE_LOG_INFO("Task manager rank {} executing query {}", _rank, query_id);
  bool const record_timing = timing_enabled();
  auto const start_ns      = record_timing ? monotonic_ns() : 0;
  // Set the device on this thread before any NVTX push so callback plugins see the right device.
  GQE_CUDA_TRY(cudaSetDevice(_rank));
  utility::nvtx_scoped_range execute_range("execute_plan");

  // Write plans (COPY) use ALL_TO_ALL so that every rank executes the write
  // task and populates its local in-memory table cache with row group references.
  // Only rank 0 actually writes data to shared memory (guarded by nvshmem_rank_zero()),
  // but all ranks create column_base references pointing to the shared objects.
  // Query plans use ROUND_ROBIN to distribute work across ranks.
  bool const is_write = plan->type() == physical::relation::relation_type::write;

  if (_multi_process) {
    auto* mp_ctx = dynamic_cast<multi_process_task_manager_context*>(_ctx);
    if (mp_ctx) {
      mp_ctx->update_scheduler(is_write ? SCHEDULER_TYPE::ALL_TO_ALL : SCHEDULER_TYPE::ROUND_ROBIN);
    }
  }

  _cache.update(std::move(descriptors));

  if (_multi_process) {
    // FIXME: Multi-GPU task migration doesn't work with zero-copy enabled.
    opt_params.read_use_zero_copy = false;
  }
  gqe::query_context query_ctx{query_id, opt_params, gqe::optimizer::optimization_configuration{}};
  gqe::context_reference ctx_ref{_ctx, &query_ctx};

  // RAII guard to unregister all tasks for this query on scope exit (success or exception).
  struct query_registration_guard {
    task_migration_service* service;
    utility::uuid query_id;
    ~query_registration_guard()
    {
      if (service) service->unregister_query(query_id);
    }
  };

  task_migration_service* migration_svc = nullptr;
  if (_multi_process) {
    auto* mp_ctx = dynamic_cast<multi_process_task_manager_context*>(_ctx);
    if (mp_ctx) migration_svc = mp_ctx->migration_service.get();
  }
  query_registration_guard guard{migration_svc, query_id};

  // flowcsd spill mode: the spill I/O (registered staging buffers and I/O threads) shared by every
  // operator of the query, and one `flowcsd_spill` line per query with the spill counters. The
  // guard is declared before the task graph, so it releases the I/O and logs after the graph (and
  // every spill file) is gone, also when the query fails.
  gqe::reset_spill_counters();
  struct spill_query_guard {
    int rank;
    utility::uuid query_id;
    std::shared_ptr<gqe::spill_io> io;
    ~spill_query_guard()
    {
      io.reset();
      gqe::log_spill_counters(rank, query_id);
    }
  };
  spill_query_guard spill_query{
    _rank,
    query_id,
    gqe::get_spill_settings().partitions > 0 ? gqe::acquire_spill_io() : nullptr};

  std::unique_ptr<gqe::task_graph> task_graph;
  {
    utility::nvtx_scoped_range build_range("build_task_graph");
    gqe::task_graph_builder graph_builder(ctx_ref, &_cache);
    task_graph = graph_builder.build(plan.get());
  }
  auto const build_end_ns = record_timing ? monotonic_ns() : 0;

  {
    utility::nvtx_scoped_range exec_range("execute_task_graph");
    if (_multi_process) {
      gqe::execute_task_graph_multi_process(ctx_ref, task_graph.get());
    } else {
      gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());
    }
  }
  auto const execute_end_ns = record_timing ? monotonic_ns() : 0;

  service::execution_result result;

  {
    utility::nvtx_scoped_range output_range("collect_results");
    if (is_write) {
      // Queries that write their result to a table don't return any result rows.

      if (task_graph->write_statistics) {
        // Collect and serialize write statistics from the task graph.
        auto stats             = task_graph->write_statistics->statistics();
        result.has_write_stats = true;
        result.write_stats     = rpc::serialize_table_statistics(stats);
      }
    } else {
      // Collect result rows for non-write queries.
      std::vector<cudf::table_view> result_views;
      for (auto const& root_task : task_graph->root_tasks) {
        if (root_task->result().has_value()) {
          result_views.push_back(root_task->result().value());
        }
      }

      if (!result_views.empty()) { result.arrow_ipc = serialize_to_arrow_ipc(result_views); }
    }
  }

  if (record_timing) {
    GQE_LOG_INFO(
      "flowcsd_timing task_manager rank={} query={} start_ns={} build_end_ns={} execute_end_ns={} "
      "collect_end_ns={}",
      _rank,
      query_id,
      start_ns,
      build_end_ns,
      execute_end_ns,
      monotonic_ns());
  }
  return result;
}

void plan_executor::refresh_stats(std::vector<service::stats_table_write> writes,
                                  utility::uuid query_id,
                                  std::vector<storage::descriptor> descriptors,
                                  std::uint64_t epoch)
{
  // The node manager owns the statistics tables but cannot write shared row groups itself (the
  // shared write path requires a task manager's NVSHMEM rank-0 context and CUDA stream). So it
  // projects the current statistics into Arrow batches and ships them here; rank 0 publishes them
  // into the node-created shared segment via shared_in_memory_table::replace_contents(), which
  // advances the table's storage_revision. Every other rank simply reattaches to the new contents
  // on its next read of the table, when the read's descriptor carries the bumped revision (see
  // remote_table_cache::update) — so only rank 0 writes, and the data lives once in shared memory.
  //
  // (void)epoch: the previous implementation mixed `epoch` into per-refresh shared-object names to
  // avoid collisions; replace_contents now frees the prior row group before publishing, so the
  // epoch is no longer needed for naming.
  (void)epoch;
  GQE_LOG_INFO("Task manager rank {} refreshing {} stats table(s) (refresh {})",
               _rank,
               writes.size(),
               query_id);

  // Register the statistics-table descriptors so this rank can resolve them; on non-writer ranks
  // this also ensures a subsequent read reattaches to the contents rank 0 publishes below.
  _cache.update(std::move(descriptors));

  // Only rank 0 writes the shared contents; other ranks reattach via storage_revision on read.
  if (!gqe::utility::multi_process::nvshmem_rank_zero()) { return; }

  auto stream   = cudf::get_default_stream();
  auto const mr = _ctx->get_table_memory_resource(memory_kind::boost_shared{});

  for (auto const& write : writes) {
    auto table      = _cache.get_table(write.table_name);
    auto* in_memory = dynamic_cast<storage::shared_in_memory_table*>(table.get());
    if (in_memory == nullptr) {
      throw std::runtime_error(
        std::format("Stats refresh target '{}' is not a shared in-memory table", write.table_name));
    }

    if (write.arrow_ipc.empty()) {
      // Nothing to materialize (e.g. no source tables): leave the stats table empty.
      in_memory->clear_row_groups(stream, mr);
      continue;
    }

    auto batch      = deserialize_arrow_ipc(write.arrow_ipc);
    auto cudf_table = arrow_batch_to_cudf(*batch, stream);
    // replace_contents reads from this device table; make sure it is fully materialized first.
    stream.synchronize();
    in_memory->replace_contents(cudf_table->view(), mr, stream);
    stream.synchronize();
  }
}

}  // namespace gqe::task_manager
