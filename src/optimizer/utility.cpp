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

#include "utility.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/optimizer/relation_properties.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <vector>

namespace gqe::optimizer::utility {

namespace {
using expression_t = gqe::expression::expression_type;
}  // namespace

bool is_equijoin_predicate(gqe::expression const* condition,
                           cudf::size_type const num_columns_left_child_relation,
                           std::vector<cudf::size_type>& left_key_indices,
                           std::vector<cudf::size_type>& right_key_indices)
{
  if (condition->type() == expression_t::binary_op) {
    auto binary_op_condition = static_cast<gqe::binary_op_expression const*>(condition);
    auto child_exprs         = condition->children();
    assert(child_exprs.size() == 2);

    cudf::size_type col_idx_0 = 0, col_idx_1 = 0;
    switch (binary_op_condition->binary_operator()) {
      case cudf::binary_operator::EQUAL:
        if (child_exprs[0]->type() == expression_t::column_reference &&
            child_exprs[1]->type() == expression_t::column_reference) {
          col_idx_0 =
            static_cast<gqe::column_reference_expression const*>(child_exprs[0])->column_idx();
          col_idx_1 =
            static_cast<gqe::column_reference_expression const*>(child_exprs[1])->column_idx();

          // Both columns must come from different child relations.
          if ((col_idx_0 < num_columns_left_child_relation) ==
              (col_idx_1 < num_columns_left_child_relation))
            return false;

          left_key_indices.push_back(col_idx_0 < num_columns_left_child_relation ? col_idx_0
                                                                                 : col_idx_1);
          right_key_indices.push_back(
            (col_idx_0 < num_columns_left_child_relation ? col_idx_1 : col_idx_0) -
            num_columns_left_child_relation);
        } else
          return false;
        break;
      case cudf::binary_operator::LOGICAL_AND:
      case cudf::binary_operator::NULL_LOGICAL_AND:
        if (!is_equijoin_predicate(
              child_exprs[0], num_columns_left_child_relation, left_key_indices, right_key_indices))
          return false;
        if (!is_equijoin_predicate(
              child_exprs[1], num_columns_left_child_relation, left_key_indices, right_key_indices))
          return false;
        break;
      default: return false;
    }
    return true;
  } else
    return false;
}

bool keys_cover_unique_key(relation_properties const& props,
                           std::vector<cudf::size_type> const& equi_keys)
{
  std::unordered_set<cudf::size_type> equi_set(equi_keys.begin(), equi_keys.end());
  for (auto const& key : props.unique_keys()) {
    if (std::all_of(key.begin(), key.end(), [&](auto c) { return equi_set.contains(c); }))
      return true;
  }
  return false;
}

}  // namespace gqe::optimizer::utility
