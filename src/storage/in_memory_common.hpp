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

#include <gqe/storage/in_memory.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <span>

namespace gqe::storage::detail {

/**
 * @brief Return the aggregate selected row count after validating that it fits in cuDF.
 */
[[nodiscard]] cudf::size_type checked_row_count(pruning_results_t const& pruning_results);

/**
 * @brief Evaluate conservative zone-map pruning for an in-memory row-group range.
 */
[[nodiscard]] std::unique_ptr<pruning_results_t> evaluate_in_memory_pruning(
  std::span<row_group_interface const* const> row_groups,
  std::span<cudf::size_type const> column_indexes,
  std::optional<arrow::compute::Expression> const& partial_filter,
  bool use_partition_pruning,
  cudf::size_type partition_size);

}  // namespace gqe::storage::detail
