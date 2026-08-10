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

#include <gqe/qep/operator/project.hpp>

#include <gqe/executor/eval.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {
namespace {

void collect_column_references(expression const& expression,
                               cudf::size_type num_columns,
                               std::vector<cudf::size_type>& indices)
{
  if (expression.type() == expression::expression_type::column_reference) {
    auto const index = static_cast<column_reference_expression const&>(expression).column_idx();
    GQE_EXPECTS(index >= 0 && index < num_columns,
                "qep::project_task::next: column reference is out of range");
    indices.push_back(index);
    return;
  }
  for (auto const* child : expression.children()) {
    GQE_EXPECTS(child != nullptr, "qep::project_task::next: expression contains a null child");
    collect_column_references(*child, num_columns, indices);
  }
}

}  // namespace

project_task::project_task(std::vector<std::unique_ptr<gqe::expression>> output_expressions)
  : _output_expressions(std::move(output_expressions))
{
  GQE_EXPECTS(!_output_expressions.empty(),
              "qep::project_task: output expressions must not be empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_output_expressions,
                                  [](auto const& expression) { return expression != nullptr; }),
              "qep::project_task: output expression must not be null",
              std::invalid_argument);
}

std::unique_ptr<task> project_task::clone() const
{
  return std::make_unique<project_task>(clone_expressions(_output_expressions));
}

std::optional<state_container> project_task::next(state_container_view inputs,
                                                  context_reference ctx_ref,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::project_task::next"};
  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "qep::project_task::next: query context must not be null");
  GQE_EXPECTS(!inputs.empty(), "qep::project_task::next: input container is empty");
  for (auto const& slot : inputs) {
    GQE_EXPECTS(slot != nullptr, "qep::project_task::next: input contains a null state slot");
    if (auto const* owned = std::get_if<state_kind::cudf_column>(slot.get())) {
      GQE_EXPECTS(owned->column != nullptr,
                  "qep::project_task::next: input contains a null owned column");
    }
  }

  auto const masked = masked_table_view::try_from(inputs);
  cudf::table_view input_table;
  std::size_t column_state_offset = 0;
  if (masked) {
    input_table = masked->columns;
    // Expression column indices exclude the leading valid_mask state.
    column_state_offset = 1;
  } else {
    GQE_EXPECTS(std::holds_alternative<state_kind::cudf_column>(*inputs.front()) ||
                  std::holds_alternative<state_kind::cudf_column_view>(*inputs.front()),
                "qep::project_task::next: input is not a plain table");
    auto const table = to_table_view(inputs);
    GQE_EXPECTS(table.has_value() && table->num_columns() > 0,
                "qep::project_task::next: input is not a non-empty plain-table shape");
    input_table = *table;
  }

  std::vector<cudf::size_type> injected_indices;
  for (auto const& expression : _output_expressions) {
    if (expression->type() == expression::expression_type::column_reference) {
      auto const index = static_cast<column_reference_expression const&>(*expression).column_idx();
      GQE_EXPECTS(index >= 0 && index < input_table.num_columns(),
                  "qep::project_task::next: column reference is out of range");
    } else {
      collect_column_references(*expression, input_table.num_columns(), injected_indices);
    }
  }
  // A column may be referenced by multiple projection expressions. Deduplicate the indices before
  // creating the temporary column views.
  std::ranges::sort(injected_indices);
  auto const unique_end = std::ranges::unique(injected_indices).begin();
  injected_indices.erase(unique_end, injected_indices.end());

  std::optional<null_mask_injection> injection;
  auto evaluation_table = input_table;
  if (masked && !injected_indices.empty()) {
    // The generated null masks back temporary input views used by expression evaluation.
    injection = inject_valid_mask_as_nulls(masked->mask, input_table, injected_indices, stream, mr);
    evaluation_table = injection->as_table_view();
  }

  auto evaluated = evaluate_expressions(ctx_ref._query_context->parameters,
                                        evaluation_table,
                                        utility::to_const_raw_ptrs(_output_expressions),
                                        0,
                                        stream,
                                        mr);
  state_container_builder builder;
  if (masked) { builder.add_state(inputs.front()); }

  for (std::size_t i = 0; i < _output_expressions.size(); ++i) {
    auto const& expression = _output_expressions[i];
    if (expression->type() == expression::expression_type::column_reference) {
      auto const index = static_cast<column_reference_expression const&>(*expression).column_idx();
      builder.add_state(inputs[column_state_offset + static_cast<std::size_t>(index)]);
    } else {
      builder.add_state(state_kind::cudf_column{std::move(evaluated.owned_result[i])});
    }
  }
  return builder.build();
}

}  // namespace qep
}  // namespace gqe
