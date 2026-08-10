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

#include <gqe/qep/shapes/indirect_table.hpp>

#include <gqe/utility/error.hpp>

#include <cudf/types.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

indirect_table_source_state::indirect_table_source_state(
  detail::passkey<indirect_table_source_state>, state_container columns)
  : _columns{std::move(columns)}
{
  GQE_EXPECTS(!_columns.empty(),
              "indirect_table_source_state: source must contain at least one column",
              std::invalid_argument);
  GQE_EXPECTS(
    _columns.size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "indirect_table_source_state: source column count exceeds cudf::size_type",
    std::invalid_argument);
  _num_columns = static_cast<cudf::size_type>(_columns.size());

  std::optional<cudf::size_type> row_count;
  for (auto const& column : _columns) {
    // A null buried in this container is malformed state (logic_error from the recognizer);
    // direct null handle arguments to sibling constructors are invalid_argument.
    auto const view = to_column_view(column);
    GQE_EXPECTS(view.has_value(),
                "indirect_table_source_state: state must be a cudf_column or cudf_column_view",
                std::invalid_argument);
    if (!row_count.has_value()) {
      row_count = view->size();
    } else {
      GQE_EXPECTS(*row_count == view->size(),
                  "indirect_table_source_state: source column row counts do not match",
                  std::invalid_argument);
    }
  }
  _row_count = *row_count;
}

cudf::table_view indirect_table_source_state::contiguous_table_view() const
{
  // The constructor validates this shape; guard against contract-violating mutation of the
  // retained states.
  auto table = to_table_view(_columns);
  GQE_EXPECTS(table.has_value(),
              "indirect_table_source_state::contiguous_table_view: failed to produce a concrete "
              "table");
  return *table;
}

indirect_table_source indirect_table_source_state::make(state_container columns)
{
  return std::make_shared<indirect_table_source_state const>(
    detail::passkey<indirect_table_source_state>{}, std::move(columns));
}

indirect_gather_source_state::indirect_gather_source_state(
  detail::passkey<indirect_gather_source_state>,
  indirect_table_source relation,
  position_list_handle positions,
  cudf::out_of_bounds_policy policy)
  : _relation{std::move(relation)}, _positions{std::move(positions)}, _policy{policy}
{
  GQE_EXPECTS(_relation != nullptr,
              "indirect_gather_source_state: relation must not be null",
              std::invalid_argument);
  GQE_EXPECTS(_positions != nullptr,
              "indirect_gather_source_state: positions must not be null",
              std::invalid_argument);
}

indirect_gather_source indirect_gather_source_state::make(indirect_table_source relation,
                                                          position_list_handle positions,
                                                          cudf::out_of_bounds_policy policy)
{
  return std::make_shared<indirect_gather_source_state const>(
    detail::passkey<indirect_gather_source_state>{},
    std::move(relation),
    std::move(positions),
    policy);
}

indirect_column_state::indirect_column_state(detail::passkey<indirect_column_state>,
                                             indirect_gather_source gather_source,
                                             cudf::size_type source_column_index)
  : _gather_source{std::move(gather_source)}, _source_column_index{source_column_index}
{
  GQE_EXPECTS(_gather_source != nullptr,
              "indirect_column_state: gather source must not be null",
              std::invalid_argument);
  GQE_EXPECTS(_source_column_index >= 0 &&
                _source_column_index < _gather_source->relation_source()->num_columns(),
              "indirect_column_state: source column index is out of range",
              std::invalid_argument);
}

shared_state indirect_column_state::make(indirect_gather_source gather_source,
                                         cudf::size_type source_column_index)
{
  auto state = std::make_shared<indirect_column_state const>(
    detail::passkey<indirect_column_state>{}, std::move(gather_source), source_column_index);
  return make_shared_state(state_kind::indirect_column{std::move(state)});
}

std::optional<indirect_table_view> indirect_table_view::try_from(state_container_view container)
{
  if (container.empty()) { return std::nullopt; }
  GQE_EXPECTS(std::ranges::none_of(container, [](auto const& slot) { return slot == nullptr; }),
              "indirect_table_view::try_from: container has a null state slot");

  auto const* mask_state         = std::get_if<state_kind::valid_mask>(container.front().get());
  std::size_t const first_column = mask_state == nullptr ? 0 : 1;
  if (first_column == container.size()) {
    return std::nullopt;  // Let masked_table_view diagnose a lone valid_mask.
  }

  auto const logical_slots = container.subspan(first_column);
  bool const claims_shape  = std::ranges::any_of(logical_slots, [](auto const& slot) {
    return std::holds_alternative<state_kind::indirect_column>(*slot);
  });
  if (!claims_shape) { return std::nullopt; }

  std::optional<cudf::column_view> mask;
  std::optional<cudf::size_type> row_count;
  if (mask_state != nullptr) {
    mask      = mask_state->column().view();
    row_count = mask->size();
  }

  indirect_table_view result;
  result.mask = mask;
  result.columns.reserve(logical_slots.size());

  for (std::size_t output_index = 0; output_index < logical_slots.size(); ++output_index) {
    auto const& state = logical_slots[output_index];
    result.columns.push_back(state);

    cudf::size_type column_rows;
    if (auto const* indirect = std::get_if<state_kind::indirect_column>(state.get())) {
      auto const& indirect_state     = indirect->state();
      auto const& gather_source      = indirect_state->gather_source();
      auto const source_column_index = indirect_state->source_column_index();
      column_rows                    = gather_source->position_list()->column().size();
      result.indirect_columns.push_back(
        indirect_column_view{output_index, gather_source, source_column_index});
    } else {
      auto const direct = to_column_view(state);
      GQE_EXPECTS(direct.has_value(),
                  "indirect_table_view::try_from: logical column must be a cudf_column or "
                  "cudf_column_view");
      column_rows = direct->size();
    }

    if (!row_count.has_value()) {
      row_count = column_rows;
    } else {
      GQE_EXPECTS(*row_count == column_rows,
                  "indirect_table_view::try_from: logical column row counts do not match");
    }
  }

  result.row_count = *row_count;
  return result;
}

}  // namespace qep
}  // namespace gqe
