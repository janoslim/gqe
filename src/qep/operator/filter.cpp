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

#include <gqe/qep/operator/filter.hpp>

#include <gqe/executor/eval.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

filter_task::filter_task(std::unique_ptr<gqe::expression> condition)
  : _condition(std::move(condition))
{
  GQE_EXPECTS(
    _condition != nullptr, "qep::filter_task: condition must not be null", std::invalid_argument);
}

std::unique_ptr<task> filter_task::clone() const
{
  return std::make_unique<filter_task>(_condition->clone());
}

std::optional<state_container> filter_task::next(state_container_view inputs,
                                                 context_reference ctx_ref,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::filter_task::next"};
  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "qep::filter_task::next: query context must not be null");

  auto const masked = masked_table_view::try_from(inputs);
  cudf::table_view input_table;
  std::size_t column_state_offset = 0;
  if (masked) {
    input_table         = masked->columns;
    column_state_offset = 1;
  } else {
    GQE_EXPECTS(!inputs.empty(), "qep::filter_task::next: input container is empty");
    GQE_EXPECTS(inputs.front() != nullptr,
                "qep::filter_task::next: input contains a null state slot");
    GQE_EXPECTS(std::holds_alternative<state_kind::cudf_column>(*inputs.front()) ||
                  std::holds_alternative<state_kind::cudf_column_view>(*inputs.front()),
                "qep::filter_task::next: input is not a plain table");
    auto const table = to_table_view(inputs);
    GQE_EXPECTS(table.has_value() && table->num_columns() > 0,
                "qep::filter_task::next: input is not a non-empty plain-table shape");
    input_table = *table;
  }

  // Plain input: the evaluated predicate becomes the output mask, so allocate it from the
  // caller's result resource. Masked input: the predicate is a temporary consumed by the AND
  // below, so keep it on the current device resource.
  auto const predicate_mr = masked ? cudf::get_current_device_resource_ref() : mr;
  std::vector<gqe::expression const*> const conditions{_condition.get()};
  auto predicate_result = evaluate_expressions(
    ctx_ref._query_context->parameters, input_table, conditions, 0, stream, predicate_mr);

  auto const predicate = predicate_result.evaluated_result.front();
  GQE_EXPECTS(predicate.type().id() == cudf::type_id::BOOL8,
              "qep::filter_task::next: condition must evaluate to BOOL8");
  GQE_EXPECTS(predicate.size() == input_table.num_rows(),
              "qep::filter_task::next: condition row count must match input table");

  std::unique_ptr<cudf::column> output_mask;
  if (masked) {
    // Compose with the upstream mask directly rather than injecting it as column nullness, so a
    // null-sensitive predicate (e.g. IS NULL) cannot resurrect an upstream-rejected row. The
    // three-valued AND may leave nulls in the output mask (null AND true = null); mask consumers
    // treat a null entry as invalid, same as false.
    output_mask = cudf::binary_operation(masked->mask,
                                         predicate,
                                         cudf::binary_operator::NULL_LOGICAL_AND,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr);
  } else if (_condition->type() == expression::expression_type::column_reference) {
    // The evaluator returns a borrowed view for a column reference.
    // Copy it because the filter output must own the valid-mask column.
    output_mask = std::make_unique<cudf::column>(predicate, stream, mr);
  } else {
    output_mask = std::move(predicate_result.owned_result.front());
  }

  GQE_LOG_TRACE(
    "Execute QEP filter task: input_rows={}, input_cols={}, input_masked={}, "
    "output_valid_mask_nulls={}",
    input_table.num_rows(),
    input_table.num_columns(),
    masked.has_value(),
    output_mask->null_count());

  // Pass through each data column's original `shared_state` so the output co-owns the upstream
  // cell — this preserves its owned/borrowed kind and avoids dangling views once the caller
  // releases its input. For masked input, skip the upstream mask slot: `output_mask` supersedes
  // it.
  state_container_builder builder;
  builder.add_state(state_kind::valid_mask{std::move(output_mask)});
  for (std::size_t i = column_state_offset; i < inputs.size(); ++i) {
    builder.add_state(inputs[i]);
  }
  return builder.build();
}

}  // namespace qep
}  // namespace gqe
