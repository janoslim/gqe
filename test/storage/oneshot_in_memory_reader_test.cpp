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

#include <gqe/storage/oneshot_in_memory_reader.hpp>

#include "../utility.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/concatenate.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

gqe::qep::state_container execute_once(
  gqe::qep::iterate_task const& task,
  gqe::context_reference ctx_ref,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref, stream, mr);
  auto output   = task.next(gqe::qep::state_container_view{iterator}, ctx_ref, stream, mr);
  if (!output) { throw std::logic_error("one-shot task emitted no output"); }
  if (task.next(gqe::qep::state_container_view{iterator}, ctx_ref, stream, mr)) {
    throw std::logic_error("one-shot task emitted more than one output");
  }
  task.finalize(std::move(iterator), ctx_ref, stream, mr);
  return std::move(*output);
}

}  // namespace

class OneshotInMemoryReaderTest : public gqe::test::BaseFixture {
 protected:
  using row_group_storage = std::deque<gqe::storage::row_group>;

  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }

  std::vector<gqe::storage::row_group_interface const*> row_group_refs(
    row_group_storage const& row_groups)
  {
    std::vector<gqe::storage::row_group_interface const*> refs;
    refs.reserve(row_groups.size());
    for (auto const& row_group : row_groups) {
      refs.push_back(&row_group);
    }
    return refs;
  }

  void append_row_group(row_group_storage& row_groups,
                        std::vector<std::vector<int32_t>> const& values,
                        std::optional<cudf::size_type> zone_map_partition_size = std::nullopt)
  {
    if (values.empty()) { throw std::invalid_argument("row group must have at least one column"); }
    auto const row_count = values.front().size();
    for (auto const& column : values) {
      if (column.size() != row_count) {
        throw std::invalid_argument("row-group columns must have equal lengths");
      }
    }

    std::vector<std::unique_ptr<cudf::column>> cudf_columns;
    cudf_columns.reserve(values.size());
    for (auto const& column : values) {
      cudf::test::fixed_width_column_wrapper<int32_t> wrapper(column.begin(), column.end());
      cudf_columns.push_back(wrapper.release());
    }

    std::vector<cudf::column_view> column_views;
    column_views.reserve(cudf_columns.size());
    for (auto const& column : cudf_columns) {
      column_views.push_back(column->view());
    }

    std::unique_ptr<gqe::zone_map> zone_map;
    if (zone_map_partition_size) {
      zone_map = std::make_unique<gqe::local_zone_map>(cudf::table_view{column_views},
                                                       *zone_map_partition_size);
    }

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    columns.reserve(cudf_columns.size());
    for (auto& column : cudf_columns) {
      columns.push_back(std::make_unique<gqe::storage::contiguous_column>(std::move(*column)));
    }
    row_groups.emplace_back(std::move(columns), std::move(zone_map));
  }

  void append_row_group_with_nullable_column(row_group_storage& row_groups,
                                             std::vector<int32_t> const& values,
                                             std::vector<bool> const& validity)
  {
    if (values.size() != validity.size()) {
      throw std::invalid_argument("column values and validity must have equal lengths");
    }

    cudf::test::fixed_width_column_wrapper<int32_t> nullable_column(
      values.begin(), values.end(), validity.begin());
    cudf::test::fixed_width_column_wrapper<int32_t> nonnullable_column(values.begin(),
                                                                       values.end());

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    columns.push_back(
      std::make_unique<gqe::storage::contiguous_column>(std::move(*nullable_column.release())));
    columns.push_back(
      std::make_unique<gqe::storage::contiguous_column>(std::move(*nonnullable_column.release())));
    row_groups.emplace_back(std::move(columns));
  }

  void append_compressed_row_group(row_group_storage& row_groups,
                                   std::vector<int32_t> const& values)
  {
    cudf::test::fixed_width_column_wrapper<int32_t> wrapper(values.begin(), values.end());
    auto cudf_column = wrapper.release();
    auto compression_config =
      gqe::compression::configuration{.primary_compression_format   = gqe::compression_format::lz4,
                                      .secondary_compression_format = gqe::compression_format::none,
                                      .compression_chunk_size       = 64 * 1024,
                                      .compression_ratio_threshold  = 1000.0,
                                      .secondary_compression_ratio_threshold      = 1000.0,
                                      .secondary_compression_multiplier_threshold = 1.0,
                                      .use_cpu_compression                        = false,
                                      .compression_level                          = 0};

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    columns.push_back(std::make_unique<gqe::storage::compressed_sliced_column>(
      std::move(*cudf_column),
      32,
      gqe::memory_kind::device{rmm::get_current_cuda_device()},
      std::move(compression_config),
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource_ref()));
    row_groups.emplace_back(std::move(columns));
  }

  std::pair<row_group_storage, std::vector<int32_t>> make_unequal_five_group_source()
  {
    row_group_storage row_groups;
    std::vector<int32_t> all_values;
    int32_t next_value = 0;
    for (cudf::size_type row_group_idx = 0; row_group_idx < 5; ++row_group_idx) {
      std::vector<int32_t> values(static_cast<std::size_t>(row_group_idx + 1));
      std::iota(values.begin(), values.end(), next_value);
      next_value += static_cast<int32_t>(values.size());
      all_values.insert(all_values.end(), values.begin(), values.end());
      append_row_group(row_groups, {std::move(values)});
    }
    return {std::move(row_groups), std::move(all_values)};
  }
};

TEST_F(OneshotInMemoryReaderTest, ProjectsMultipleRowGroupsOnExplicitStreamAndResource)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2}, {10, 20}});
  append_row_group(row_groups, {{3}, {30}});

  gqe::storage::oneshot_in_memory_read_task task{
    row_group_refs(row_groups),
    {1, 0},
    {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}}};
  rmm::cuda_stream stream;
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  auto result = execute_once(task, ctx_ref(), stream.view(), result_mr);
  stream.synchronize();

  auto result_table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(result_table);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_value({10, 20, 30});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_before({1, 2, 3});
  auto expected = make_table_from_columns(expected_value.release(), expected_before.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *result_table);
  for (auto const& column : *result_table) {
    EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(const_cast<void*>(column.head())));
  }
}

TEST_F(OneshotInMemoryReaderTest, SplitIsIdempotentAndIncludesNaturallyEmptyTail)
{
  auto [row_groups, all_values] = make_unequal_five_group_source();
  gqe::storage::oneshot_in_memory_read_task task{
    row_group_refs(row_groups), {0}, {cudf::data_type{cudf::type_id::INT32}}};
  ASSERT_EQ(task.max_splits(), 5);

  constexpr std::array<cudf::size_type, 4> expected_rows{3, 7, 5, 0};
  constexpr std::array<cudf::size_type, 4> split_order{3, 0, 2, 1};
  std::array<gqe::qep::state_container, 4> outputs;
  for (auto const split_idx : split_order) {
    auto split                                   = task.split(split_idx, 4);
    outputs[static_cast<std::size_t>(split_idx)] = execute_once(*split, ctx_ref());
    EXPECT_EQ(gqe::qep::get_row_count(outputs[static_cast<std::size_t>(split_idx)]),
              expected_rows[static_cast<std::size_t>(split_idx)]);
    EXPECT_EQ(task.max_splits(), 5);
  }

  auto repeated_split = task.split(1, 4);
  auto repeated       = execute_once(*repeated_split, ctx_ref());
  auto repeated_table = gqe::qep::to_table_view(repeated);
  auto first_table    = gqe::qep::to_table_view(outputs[1]);
  ASSERT_TRUE(repeated_table);
  ASSERT_TRUE(first_table);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*first_table, *repeated_table);

  std::vector<cudf::table_view> split_views;
  split_views.reserve(outputs.size());
  for (auto const& output : outputs) {
    auto table_view = gqe::qep::to_table_view(output);
    ASSERT_TRUE(table_view);
    ASSERT_EQ(table_view->num_columns(), 1);
    EXPECT_EQ(table_view->column(0).type(), cudf::data_type{cudf::type_id::INT32});
    split_views.push_back(*table_view);
  }
  auto combined = cudf::concatenate(split_views);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values(all_values.begin(),
                                                                  all_values.end());
  auto expected = make_table_from_columns(expected_values.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), combined->view());
}

TEST_F(OneshotInMemoryReaderTest, CountOnlyAcrossSplitsAndEmptyInput)
{
  [[maybe_unused]] auto [row_groups, all_values] = make_unequal_five_group_source();
  gqe::storage::oneshot_in_memory_read_task task{row_group_refs(row_groups), {}, {}};

  constexpr std::array<cudf::size_type, 4> expected_rows{3, 7, 5, 0};
  cudf::size_type total_rows = 0;
  for (cudf::size_type split_idx = 0; split_idx < 4; ++split_idx) {
    auto split      = task.split(split_idx, 4);
    auto output     = execute_once(*split, ctx_ref());
    auto const rows = gqe::qep::get_row_count(output);
    EXPECT_EQ(rows, expected_rows[static_cast<std::size_t>(split_idx)]);
    total_rows += rows;
  }
  EXPECT_EQ(total_rows, 15);

  row_group_storage empty_row_groups;
  gqe::storage::oneshot_in_memory_read_task empty_task{row_group_refs(empty_row_groups), {}, {}};
  ASSERT_EQ(empty_task.max_splits(), 1);
  auto empty_split  = empty_task.split(0, 1);
  auto empty_output = execute_once(*empty_split, ctx_ref());
  EXPECT_EQ(gqe::qep::get_row_count(empty_output), 0);
}

TEST_F(OneshotInMemoryReaderTest, PruningConfigurationIsPreservedBySplit)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;

  row_group_storage row_groups;
  append_row_group(row_groups, {{0, 1}}, 2);
  append_row_group(row_groups, {{10, 11}}, 2);
  append_row_group(row_groups, {{20, 21}}, 2);
  append_row_group(row_groups, {{30, 31}}, 2);

  auto filter = std::make_unique<gqe::greater_expression>(
    std::make_unique<gqe::column_reference_expression>(0),
    std::make_unique<gqe::literal_expression<int32_t>>(15));
  gqe::storage::oneshot_in_memory_read_task task{
    row_group_refs(row_groups), {0}, {cudf::data_type{cudf::type_id::INT32}}, std::move(filter)};

  auto pruned_split  = task.split(0, 4);
  auto pruned_output = execute_once(*pruned_split, ctx_ref());
  EXPECT_EQ(gqe::qep::get_row_count(pruned_output), 0);

  auto retained_split  = task.split(2, 4);
  auto retained_output = execute_once(*retained_split, ctx_ref());
  auto retained_table  = gqe::qep::to_table_view(retained_output);
  ASSERT_TRUE(retained_table);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values({20, 21});
  auto expected = make_table_from_columns(expected_values.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *retained_table);
}

TEST_F(OneshotInMemoryReaderTest, ReadableViewConstructsTaskWithProjectionAndFilter)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;

  row_group_storage row_groups;
  append_row_group(row_groups, {{0, 1}, {10, 11}}, 2);
  append_row_group(row_groups, {{20, 21}, {30, 31}}, 2);

  gqe::storage::in_memory_table table{
    gqe::memory_kind::device{rmm::get_current_cuda_device()},
    {"value", "projected"},
    {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
    get_task_manager_ctx()};
  auto appender = table.get_row_group_appender();
  while (!row_groups.empty()) {
    appender(std::move(row_groups.front()));
    row_groups.pop_front();
  }

  auto filter = std::make_unique<gqe::greater_expression>(
    std::make_unique<gqe::column_reference_expression>(0),
    std::make_unique<gqe::literal_expression<int32_t>>(15));
  auto task = table.readable_view()->get_iterate_task(
    {"projected", "value"},
    {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
    std::nullopt,
    std::move(filter));
  auto output       = execute_once(*task, ctx_ref());
  auto output_table = gqe::qep::to_table_view(output);

  ASSERT_TRUE(output_table);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_projected({30, 31});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values({20, 21});
  auto expected = make_table_from_columns(expected_projected.release(), expected_values.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *output_table);
}

TEST_F(OneshotInMemoryReaderTest, ReadableViewValidatesArguments)
{
  gqe::storage::in_memory_table table{gqe::memory_kind::device{rmm::get_current_cuda_device()},
                                      {"value"},
                                      {cudf::data_type{cudf::type_id::INT32}},
                                      get_task_manager_ctx()};

  EXPECT_THROW(std::ignore = table.readable_view()->get_iterate_task({"value"}, {}),
               std::length_error);
  EXPECT_THROW(std::ignore = table.readable_view()->get_iterate_task(
                 {"missing"}, {cudf::data_type{cudf::type_id::INT32}}),
               std::out_of_range);
  EXPECT_THROW(std::ignore = table.readable_view()->get_iterate_task(
                 {"value"}, {cudf::data_type{cudf::type_id::INT32}}, cudf::size_type{0}),
               std::logic_error);

  auto filter = std::make_unique<gqe::greater_expression>(
    std::make_unique<gqe::column_reference_expression>(0),
    std::make_unique<gqe::literal_expression<int32_t>>(0));
  EXPECT_THROW(
    std::ignore = table.readable_view()->get_iterate_task({}, {}, std::nullopt, std::move(filter)),
    std::invalid_argument);
}

TEST_F(OneshotInMemoryReaderTest, ValidatesColumnIndexesAndTypesDuringInitialize)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1}});

  gqe::storage::oneshot_in_memory_read_task invalid_column_index{
    row_group_refs(row_groups), {1}, {cudf::data_type{cudf::type_id::INT32}}};
  EXPECT_THROW(
    std::ignore = invalid_column_index.initialize(gqe::qep::state_container_view{}, ctx_ref()),
    std::out_of_range);

  gqe::storage::oneshot_in_memory_read_task wrong_type{
    row_group_refs(row_groups), {0}, {cudf::data_type{cudf::type_id::INT64}}};
  EXPECT_THROW(std::ignore = wrong_type.initialize(gqe::qep::state_container_view{}, ctx_ref()),
               std::invalid_argument);
}

TEST_F(OneshotInMemoryReaderTest, RejectsNestedOutputTypesDuringConstruction)
{
  for (auto const type :
       {cudf::data_type{cudf::type_id::LIST}, cudf::data_type{cudf::type_id::STRUCT}}) {
    EXPECT_THROW(std::ignore = gqe::storage::oneshot_in_memory_read_task({}, {0}, {type}),
                 std::invalid_argument);
  }
}

TEST_F(OneshotInMemoryReaderTest, RejectsMixedProjectedColumnLayoutsDuringInitialize)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1}});
  append_compressed_row_group(row_groups, {2});

  gqe::storage::oneshot_in_memory_read_task task{
    row_group_refs(row_groups), {0}, {cudf::data_type{cudf::type_id::INT32}}};
  EXPECT_THROW(std::ignore = task.initialize(gqe::qep::state_container_view{}, ctx_ref()),
               std::invalid_argument);
}

TEST_F(OneshotInMemoryReaderTest, PreservesNullsForContiguousStringColumns)
{
  row_group_storage row_groups;
  cudf::test::strings_column_wrapper first({"alpha", ""}, {true, false});
  cudf::test::strings_column_wrapper second({"", "gamma"}, {false, true});
  std::array<std::unique_ptr<cudf::column>, 2> input_columns{first.release(), second.release()};
  for (auto& column : input_columns) {
    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    columns.push_back(std::make_unique<gqe::storage::contiguous_column>(std::move(*column)));
    row_groups.emplace_back(std::move(columns));
  }

  gqe::storage::oneshot_in_memory_read_task task{
    row_group_refs(row_groups), {0}, {cudf::data_type{cudf::type_id::STRING}}};
  auto output       = execute_once(task, ctx_ref());
  auto output_table = gqe::qep::to_table_view(output);

  ASSERT_TRUE(output_table);
  ASSERT_EQ(output_table->num_columns(), 1);
  EXPECT_EQ(output_table->column(0).null_count(), 2);
  cudf::test::strings_column_wrapper expected({"alpha", "", "", "gamma"},
                                              {true, false, false, true});
  auto expected_table = make_table_from_columns(expected.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), *output_table);
}

TEST_F(OneshotInMemoryReaderTest, RejectsNullableFixedWidthProjectionButAllowsUnprojectedNulls)
{
  row_group_storage row_groups;
  append_row_group_with_nullable_column(row_groups, {1, 2, 3}, {true, false, true});

  gqe::storage::oneshot_in_memory_read_task nullable_projection{
    row_group_refs(row_groups), {0}, {cudf::data_type{cudf::type_id::INT32}}};
  EXPECT_THROW(
    std::ignore = nullable_projection.initialize(gqe::qep::state_container_view{}, ctx_ref()),
    std::invalid_argument);

  gqe::storage::oneshot_in_memory_read_task nonnullable_projection{
    row_group_refs(row_groups), {1}, {cudf::data_type{cudf::type_id::INT32}}};
  EXPECT_EQ(gqe::qep::get_row_count(execute_once(nonnullable_projection, ctx_ref())), 3);

  gqe::storage::oneshot_in_memory_read_task count_only{row_group_refs(row_groups), {}, {}};
  EXPECT_EQ(gqe::qep::get_row_count(execute_once(count_only, ctx_ref())), 3);
}
