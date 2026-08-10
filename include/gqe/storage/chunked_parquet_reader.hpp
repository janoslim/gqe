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

#pragma once

#include <gqe/context_reference.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/storage/hive_partitioning.hpp>

#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <vector>

namespace gqe {
namespace storage {

/**
 * @brief Streaming Parquet read task.
 *
 * Wraps `cudf::io::chunked_parquet_reader` as a `qep::iterate_task`. Each
 * call to `next` returns one chunk's worth of data, type-enforced to the
 * requested data types. Returns `std::nullopt` to signal end-of-stream.
 *
 * # Output Shape
 *
 * Chunks with projected columns are emitted as `[cudf_column, ...]`.
 *
 * # Count-Star Mode
 *
 * When `column_names` is empty, the task does not materialize any Parquet columns. Instead,
 * it reads the dataset row count from Parquet metadata and emits a single chunk shaped
 * `[row_count{N}]`, where `N` is the total row count. Downstream operators must read this
 * value from the `row_count` state slot rather than from `cudf::table::num_rows()`.
 *
 * # Hive Partition Pruning
 *
 * Pass `hive_pruner_options` to enable pruning for Hive-partitioned datasets. The
 * partition column value is encoded in the directory name, e.g.
 * `lineitem/l_shipmode=1/part-0001.parquet`, and is absent from the on-disk Parquet schema.
 *
 * The reader parses each file path for `<column_name>=<value>`, reads the allowed values from
 * the predecessor input state, drops files whose parsed key is not present in that haystack,
 * omits the partition column from the Parquet read, and synthesizes the partition column per
 * emitted chunk.
 *
 * The haystack input must be a single integral column. Files that do not match the Hive path
 * pattern are rejected in strict mode; Hive default-partition entries are treated as null and
 * dropped.
 */
class chunked_parquet_read_task : public qep::iterate_task {
 public:
  /**
   * @brief Construct a Parquet read task.
   *
   * @param[in] file_paths Files to read.
   * @param[in] column_names Columns to load, in output order. Empty enables count-star mode.
   * @param[in] output_data_types Expected data type per column. A column
   *   whose on-disk type differs is cast to the expected type. Must have
   *   the same length as `column_names`.
   * @param[in] pruner When set, enables Hive-partition pruning. The
   *   predecessor task's output supplies the in-list haystack (a single
   *   integral-typed column).
   */
  chunked_parquet_read_task(std::vector<std::string> file_paths,
                            std::vector<std::string> column_names,
                            std::vector<cudf::data_type> output_data_types,
                            std::optional<hive_pruner_options> pruner = std::nullopt);

  /**
   * @copydoc gqe::qep::iterate_task::initialize
   */
  [[nodiscard]] qep::state_container initialize(
    qep::state_container_view inputs,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::iterate_task::next
   */
  [[nodiscard]] std::optional<qep::state_container> next(
    qep::state_container_view iterator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::iterate_task::finalize
   */
  void finalize(
    qep::state_container&& iterator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<qep::task> clone() const override;

 private:
  std::vector<std::string> _file_paths;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _output_data_types;
  std::optional<hive_pruner_options> _pruner;
};

}  // namespace storage
}  // namespace gqe
