/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cudf/types.hpp>

#include <cassert>
#include <vector>

namespace gqe {
class expression;

namespace optimizer {

class relation_properties;

namespace utility {

/**
 * @brief This function determine whether each projection index is from the left or right child.
 * Then offset it accordingly assuming a left/right swap.
 *
 * @param projection_indices List of projection indices to offset
 * @param n_cols_left Number of columns from left child
 * @param n_cols_right Number of columns from right child
 */
inline void swap_projection_indices_inplace(std::vector<cudf::size_type>& projection_indices,
                                            cudf::size_type n_cols_left,
                                            cudf::size_type n_cols_right)
{
  for (auto& projection_index : projection_indices) {
    assert(projection_index < (n_cols_left + n_cols_right));
    projection_index = projection_index < n_cols_left ? projection_index + n_cols_right
                                                      : projection_index - n_cols_left;
  }
}

/**
 * @brief Check whether @p condition is a conjunction of EQUAL comparisons between one column from
 * each child relation, and collect the per-side key column indices.
 *
 * NULL_EQUALS (IS NOT DISTINCT FROM) is intentionally excluded: a nullable UNIQUE column permits
 * multiple NULL rows, so treating NULL-equal keys as unique could map distinct NULL rows to the
 * same slot and produce incorrect results.
 *
 * @param[in]  condition                        Join predicate to inspect.
 * @param[in]  num_columns_left_child_relation  Output column count of the left child, used to split
 *                                              global column indices into per-side indices.
 * @param[out] left_key_indices                 Collected left-side key column indices.
 * @param[out] right_key_indices                Collected right-side key column indices (relative to
 *                                              the right child).
 * @return true if the predicate is a valid equi-join predicate; false otherwise.
 */
bool is_equijoin_predicate(gqe::expression const* condition,
                           cudf::size_type num_columns_left_child_relation,
                           std::vector<cudf::size_type>& left_key_indices,
                           std::vector<cudf::size_type>& right_key_indices);

/**
 * @brief Return true if any unique key-set registered on @p props is fully covered by @p equi_keys.
 *
 * A registered key-set is covered iff every one of its columns appears in the equi-join keys, i.e.
 * the join keys are at least as restrictive as that unique key. A singleton key {col} is covered
 * iff col is among the equi-join keys.
 *
 * @param props     Relation properties carrying the registered unique key-sets.
 * @param equi_keys Per-side equi-join key column indices.
 */
bool keys_cover_unique_key(relation_properties const& props,
                           std::vector<cudf::size_type> const& equi_keys);

}  // namespace utility
}  // namespace optimizer
}  // namespace gqe
