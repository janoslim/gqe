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

#include <gqe/physical/relation.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/storage/table_provider.hpp>

namespace gqe {
namespace qep {

/**
 * @brief Transformation pass that lowers a physical plan to a Query Execution Plan.
 *
 * Lowers a `physical::relation` tree to a `query_execution_plan`. Each physical relation is
 * lowered to one or more QEP tasks. The lowering ensures that physical plan nodes reached
 * via multiple dependency paths are lowered exactly once.
 *
 * The lowering rejects inputs with `std::logic_error` in the following cases:
 *
 * - Relations that lack QEP operator support.
 * - Conditions containing `subquery_expression`. Scalar subqueries are assumed to be
 *   unnested into joins by the optimizer.
 * - Aggregate key or value expressions that are not bare column references. Run the
 *   `complex_expression_extraction_into_project` optimizer rule to materialize computed
 *   grouping keys and measures upstream.
 *
 * # Aggregation Functions
 *
 * Aggregation functions must be distributive. Non-distributive functions (i.e., algebraic
 * and holistic functions, per Gray's classification) are rejected during lowering or by the
 * aggregate and reduce operators. Run an optimizer rule (e.g., `mean_decomposition`) to transform
 * them into distributive functions.
 *
 * # Masked outputs
 *
 * Some QEP operators emit (or pass through) a masked-table state container (`[valid_mask,
 * columns...]`). The transformation tracks the masked shape through the lowering and injects
 * `gather_task`s where a consumer requires a regular-table state container.
 *
 * `filter_task`, `reduce_aggregate_task`, and `perfect_hash_partial_aggregate_task` accept
 * masked-table state containers, so the lowering connects them directly without an intervening
 * `gather_task`. Their operator-specific validation requirements still apply.
 *
 * The final result must be a regular-table state container.
 *
 * @param[in] root Root of the physical plan tree.
 * @param[in] table_provider Resolves table-name references in the physical plan to
 *   `storage::table` instances. Must outlive the call.
 *
 * @return The lowered QEP.
 */
[[nodiscard]] query_execution_plan physical_qep_transformation(
  physical::relation* root, storage::table_provider const* table_provider);

}  // namespace qep
}  // namespace gqe
