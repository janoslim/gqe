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

#include <gqe/qep/operator/gather.hpp>

#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

namespace {

struct gather_group {
  indirect_gather_source gather_source;
  std::vector<std::size_t> output_indexes;
  std::vector<cudf::size_type> source_column_indexes;
};

struct compacted_position_list {
  position_list_handle original_state;
  position_list_handle compacted;
};

[[nodiscard]] state_container gather_indirect(indirect_table_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  std::vector<gather_group> groups;
  for (auto const& indirect : input.indirect_columns) {
    auto group = std::ranges::find_if(groups, [&](auto const& candidate) {
      return candidate.gather_source.get() == indirect.gather_source.get();
    });
    if (group == groups.end()) {
      groups.push_back(gather_group{
        indirect.gather_source, {indirect.output_index}, {indirect.source_column_index}});
    } else {
      group->output_indexes.push_back(indirect.output_index);
      group->source_column_indexes.push_back(indirect.source_column_index);
    }
  }

  std::vector<compacted_position_list> compacted_maps;
  if (input.mask.has_value()) {
    for (auto const& group : groups) {
      auto const& position_list = group.gather_source->position_list();
      auto already_compacted    = std::ranges::find_if(compacted_maps, [&](auto const& candidate) {
        return candidate.original_state.get() == position_list.get();
      });
      if (already_compacted != compacted_maps.end()) { continue; }

      compacted_maps.push_back(compacted_position_list{
        position_list, compact_position_list(position_list, *input.mask, stream, mr)});
    }
  }

  state_container output(input.columns.size());
  for (auto const& group : groups) {
    auto const& position_list = group.gather_source->position_list();
    position_list_handle positions;
    if (input.mask.has_value()) {
      auto const compacted = std::ranges::find_if(compacted_maps, [&](auto const& candidate) {
        return candidate.original_state.get() == position_list.get();
      });
      GQE_EXPECTS(compacted != compacted_maps.end(),
                  "qep::gather_task::next: missing compacted position list");
      positions = compacted->compacted;
    } else {
      positions = position_list;
    }

    auto gathered = materialize_indirect_columns(
                      group.gather_source, group.source_column_indexes, positions, stream, mr)
                      .release();
    GQE_EXPECTS(gathered.size() == group.output_indexes.size(),
                "qep::gather_task::next: gathered group has unexpected column count");
    for (std::size_t i = 0; i < gathered.size(); ++i) {
      output[group.output_indexes[i]] =
        make_shared_state(state_kind::cudf_column{std::move(gathered[i])});
    }
  }

  std::vector<std::size_t> direct_indexes;
  std::vector<cudf::column_view> direct_columns;
  for (std::size_t i = 0; i < input.columns.size(); ++i) {
    if (std::holds_alternative<state_kind::indirect_column>(*input.columns[i])) { continue; }
    if (!input.mask.has_value()) {
      output[i] = input.columns[i];
    } else {
      direct_indexes.push_back(i);
      direct_columns.push_back(to_column_view(input.columns[i]).value());
    }
  }

  if (!direct_columns.empty()) {
    auto filtered_direct =
      cudf::apply_boolean_mask(cudf::table_view{direct_columns}, *input.mask, stream, mr)
        ->release();
    for (std::size_t i = 0; i < filtered_direct.size(); ++i) {
      output[direct_indexes[i]] =
        make_shared_state(state_kind::cudf_column{std::move(filtered_direct[i])});
    }
  }

  GQE_EXPECTS(std::ranges::none_of(output, [](auto const& slot) { return slot == nullptr; }),
              "qep::gather_task::next: failed to materialize a logical column");
  return output;
}

}  // namespace

std::unique_ptr<task> gather_task::clone() const { return std::make_unique<gather_task>(); }

std::optional<state_container> gather_task::next(state_container_view inputs,
                                                 context_reference,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::gather_task::next"};
  GQE_EXPECTS(std::ranges::none_of(inputs, [](auto const& slot) { return slot == nullptr; }),
              "qep::gather_task::next: input contains a null state slot");

  if (auto const indirect = indirect_table_view::try_from(inputs); indirect.has_value()) {
    return gather_indirect(*indirect, stream, mr);
  } else if (auto const masked = masked_table_view::try_from(inputs); masked.has_value()) {
    auto output = cudf::apply_boolean_mask(masked->columns, masked->mask, stream, mr);
    return state_container_builder().add_state(std::move(*output)).build();
  } else if (try_row_count(inputs).has_value()) {
    return make_mutable_state_copy(inputs);
  } else if (to_table_view(inputs).has_value()) {
    return make_mutable_state_copy(inputs);
  } else {
    throw std::logic_error("qep::gather_task::next: input is not a supported relational shape");
  }
}

}  // namespace qep
}  // namespace gqe
