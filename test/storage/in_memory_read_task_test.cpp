/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/device_properties.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/memory_utilities.hpp>
#include <gqe/memory_resource/shared_resource_adaptor.hpp>
#include <gqe/node_manager/context.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cuda/memory_resource>
#include <cudf/column/column.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>

namespace {

bool is_in_segment(boost::interprocess::managed_shared_memory const& segment, void const* ptr)
{
  auto const start = reinterpret_cast<std::uintptr_t>(segment.get_address());
  auto const end   = start + segment.get_size();
  auto const value = reinterpret_cast<std::uintptr_t>(ptr);
  return value >= start && value < end;
}

gqe::compression::configuration make_shared_storage_compression_config()
{
  return gqe::compression::configuration{
    .primary_compression_format                 = gqe::compression_format::lz4,
    .secondary_compression_format               = gqe::compression_format::none,
    .compression_chunk_size                     = 64 * 1024,
    .compression_ratio_threshold                = 1000.0,
    .secondary_compression_ratio_threshold      = 1000.0,
    .secondary_compression_multiplier_threshold = 1.0,
    .use_cpu_compression                        = true,
    .compression_level                          = 0,
    .decompress_backend                         = gqe::decompression_backend::auto_};
}

}  // namespace

struct test_parameters {
  test_parameters(bool read_use_zero_copy,
                  gqe::compression_format comp_format,
                  bool use_overlap_mtx = false)
    : read_use_zero_copy(read_use_zero_copy),
      comp_format(comp_format),
      use_overlap_mtx(use_overlap_mtx)
  {
  }

  bool read_use_zero_copy;
  gqe::compression_format comp_format;
  bool use_overlap_mtx;
};

class InMemoryReadTest : public gqe::test::BaseFixtureWithParam<test_parameters> {
 public:
  InMemoryReadTest()

  {
    auto const params = GetParam();

    auto opms               = gqe::make_optimization_parameters(false);
    opms.read_use_zero_copy = params.read_use_zero_copy;
    opms.compression_format = params.comp_format;
    opms.use_overlap_mtx    = params.use_overlap_mtx;

    initialize_contexts(opms);
    query_ctx = get_query_ctx();
  }

  void SetUp() override
  {
    // Setup data
    cudf::test::fixed_width_column_wrapper<int32_t> col_0({6, 5, 4, 3, 2, 1});
    cudf::test::fixed_width_column_wrapper<float> col_1({1.0, 5.0, 3.0, 9.0, 7.0, 0.0});
    std::vector<std::unique_ptr<cudf::column>> test_columns;
    test_columns.push_back(col_0.release());
    test_columns.push_back(col_1.release());

    // Setup row group
    auto comp_format                        = query_ctx->parameters.compression_format;
    auto const chunk_size                   = query_ctx->parameters.compression_chunk_size;
    auto compression_ratio_threshold        = query_ctx->parameters.compression_ratio_threshold;
    auto const secondary_compression_format = query_ctx->parameters.secondary_compression_format;
    auto const secondary_compression_ratio_threshold =
      query_ctx->parameters.secondary_compression_ratio_threshold;
    auto const secondary_compression_multiplier_threshold =
      query_ctx->parameters.secondary_compression_multiplier_threshold;
    auto const use_cpu_compression = query_ctx->parameters.use_cpu_compression;
    auto const compression_level   = query_ctx->parameters.compression_level;
    auto const decompress_backend  = query_ctx->parameters.decompression_backend;
    auto const partition_size      = query_ctx->parameters.zone_map_partition_size;
    auto const memory_kind         = gqe::memory_kind::numa_pinned{};

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    std::transform(
      test_columns.cbegin(),
      test_columns.cend(),
      std::back_inserter(columns),
      [comp_format,
       chunk_size,
       compression_ratio_threshold,
       secondary_compression_format,
       secondary_compression_ratio_threshold,
       secondary_compression_multiplier_threshold,
       use_cpu_compression,
       compression_level,
       decompress_backend,
       partition_size,
       memory_kind](auto const& col) mutable -> std::unique_ptr<gqe::storage::column_base> {
        if (comp_format == gqe::compression_format::none) {
          return std::make_unique<gqe::storage::contiguous_column>(cudf::column(*col));
        } else {
          auto cudf_col           = cudf::column(*col);
          auto compression_config = gqe::compression::configuration{
            .primary_compression_format            = comp_format,
            .secondary_compression_format          = secondary_compression_format,
            .compression_chunk_size                = chunk_size,
            .compression_ratio_threshold           = compression_ratio_threshold,
            .secondary_compression_ratio_threshold = secondary_compression_ratio_threshold,
            .secondary_compression_multiplier_threshold =
              secondary_compression_multiplier_threshold,
            .use_cpu_compression = use_cpu_compression,
            .compression_level   = compression_level,
            .decompress_backend  = decompress_backend};
          return std::make_unique<gqe::storage::compressed_sliced_column>(
            std::move(cudf_col),
            partition_size,
            memory_kind,
            std::move(compression_config),
            cudf::get_default_stream(),
            rmm::mr::get_current_device_resource_ref());
        }
      });
    gqe::storage::row_group row_group(std::move(columns));

    // Setup test table
    test_table = std::make_unique<cudf::table>(std::move(test_columns));

    // Setup table types
    std::vector<std::string> col_names     = {"int", "float"};
    std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT32),
                                              cudf::data_type(cudf::type_id::FLOAT32)};

    // Setup GQE table
    table = std::make_unique<gqe::storage::in_memory_table>(
      memory_kind, col_names, col_types, get_task_manager_ctx());

    // Add row groups to table
    table->get_row_group_appender()(std::move(row_group));
  }

  void TearDown() override { table = nullptr; }

  std::unique_ptr<gqe::storage::in_memory_table> table;
  std::unique_ptr<cudf::table> test_table;
  gqe::query_context* query_ctx;
};

TEST_P(InMemoryReadTest, ReadFirstCol)
{
  constexpr int32_t stage_id             = 0;
  const std::vector<int32_t> task_ids    = {0};
  std::vector<std::string> col_names     = {"int"};
  std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT32)};

  std::vector<gqe::storage::in_memory_readable_view::task_parameters> task_parameters;
  std::transform(
    task_ids.cbegin(),
    task_ids.cend(),
    std::back_inserter(task_parameters),
    [](auto id) -> gqe::storage::in_memory_readable_view::task_parameters { return {id}; });

  gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
  auto tasks = table->readable_view()->get_read_tasks(
    std::move(task_parameters), ctx_ref, stage_id, col_names, col_types);

  EXPECT_EQ(tasks.size(), 1);
  ASSERT_FALSE(tasks.empty());

  auto& task = tasks.at(0);
  task->execute();
  auto result = task->result();

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->column(0), test_table->get_column(0));
}

TEST_P(InMemoryReadTest, ReadSecondCol)
{
  constexpr int32_t stage_id             = 0;
  const std::vector<int32_t> task_ids    = {0};
  std::vector<std::string> col_names     = {"float"};
  std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::FLOAT32)};

  std::vector<gqe::storage::in_memory_readable_view::task_parameters> task_parameters;
  std::transform(
    task_ids.cbegin(),
    task_ids.cend(),
    std::back_inserter(task_parameters),
    [](auto id) -> gqe::storage::in_memory_readable_view::task_parameters { return {id}; });

  gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
  auto tasks = table->readable_view()->get_read_tasks(
    std::move(task_parameters), ctx_ref, stage_id, col_names, col_types);

  EXPECT_EQ(tasks.size(), 1);
  ASSERT_FALSE(tasks.empty());

  auto& task = tasks.at(0);
  task->execute();
  auto result = task->result();

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->column(0), test_table->get_column(1));
}

TEST_P(InMemoryReadTest, ReadAll)
{
  constexpr int32_t stage_id             = 0;
  const std::vector<int32_t> task_ids    = {0};
  std::vector<std::string> col_names     = {"int", "float"};
  std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT32),
                                            cudf::data_type(cudf::type_id::FLOAT32)};

  std::vector<gqe::storage::in_memory_readable_view::task_parameters> task_parameters;
  std::transform(
    task_ids.cbegin(),
    task_ids.cend(),
    std::back_inserter(task_parameters),
    [](auto id) -> gqe::storage::in_memory_readable_view::task_parameters { return {id}; });

  gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
  auto tasks = table->readable_view()->get_read_tasks(
    std::move(task_parameters), ctx_ref, stage_id, col_names, col_types);

  EXPECT_EQ(tasks.size(), 1);
  ASSERT_FALSE(tasks.empty());

  auto& task = tasks.at(0);
  task->execute();
  auto result = task->result();

  ASSERT_TRUE(result.has_value());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result, test_table->view());
}

INSTANTIATE_TEST_SUITE_P(
  ZeroCopyOnOff,
  InMemoryReadTest,
  testing::Values(test_parameters{true, gqe::compression_format::none},
                  test_parameters{false, gqe::compression_format::none},
                  test_parameters{false, gqe::compression_format::ans},
                  test_parameters{false, gqe::compression_format::lz4},
                  test_parameters{false, gqe::compression_format::snappy},
                  test_parameters{false, gqe::compression_format::gdeflate},
                  test_parameters{false, gqe::compression_format::deflate},
                  test_parameters{false, gqe::compression_format::cascaded},
                  test_parameters{false, gqe::compression_format::bitpacking},
                  test_parameters{false, gqe::compression_format::zstd},
                  test_parameters{false, gqe::compression_format::bitcomp},
                  test_parameters{false, gqe::compression_format::ans, true},
                  test_parameters{false, gqe::compression_format::none, true}));

TEST(CompressedSlicedColumnSharedMemoryTest, SegmentConstructsFixedWidthAndStringColumns)
{
  static_assert(!std::is_polymorphic_v<gqe::storage::shared_compressed_sliced_column_storage>);
  static_assert(
    !std::is_polymorphic_v<gqe::storage::shared_string_compressed_sliced_column_storage<false>>);

  constexpr char segment_name[] = "gqe_compressed_sliced_column_allocator_test";
  boost::interprocess::shared_memory_object::remove(segment_name);
  {
    [[maybe_unused]] boost::interprocess::managed_shared_memory created_segment(
      boost::interprocess::create_only, segment_name, 8 << 20);

    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only,
                                                       segment_name);
    auto shared_mr = std::make_shared<gqe::memory_resource::boost_shared_memory_resource>(&segment);
    auto shared_mr_ref = gqe::memory_resource::shared_resource_adaptor{shared_mr};
    gqe::shared_memory_allocator alloc(segment.get_segment_manager());
    auto stream = cudf::get_default_stream();
    auto kind   = gqe::memory_kind::boost_shared{};

    cudf::test::fixed_width_column_wrapper<int32_t> fixed_width_values({1, 2, 3, 4});
    auto* fixed_width_storage =
      segment.construct<gqe::storage::shared_compressed_sliced_column_storage>(
        "fixed_width_column")(cudf::column(fixed_width_values),
                              2,
                              kind,
                              make_shared_storage_compression_config(),
                              stream,
                              shared_mr_ref,
                              alloc);
    ASSERT_NE(fixed_width_storage, nullptr);
    auto fixed_width_handle = segment.get_handle_from_address(fixed_width_storage);
    gqe::storage::shared_compressed_sliced_column fixed_width_column(segment, fixed_width_handle);
    gqe::storage::compressed_sliced_column_interface& fixed_width_interface = fixed_width_column;

    EXPECT_EQ(fixed_width_interface.size(), 4);
    auto fixed_width_partition = fixed_width_interface.data_buffer_view().get_partition(0);
    EXPECT_GT(fixed_width_partition.size(), 0);
    EXPECT_TRUE(is_in_segment(segment, fixed_width_partition.data()));

    cudf::test::strings_column_wrapper string_values({"aa", "bbb", "c", "dddd"});
    auto* string_storage =
      segment.construct<gqe::storage::shared_string_compressed_sliced_column_storage<false>>(
        "string_column")(cudf::column(string_values),
                         2,
                         kind,
                         make_shared_storage_compression_config(),
                         stream,
                         shared_mr_ref,
                         alloc);
    ASSERT_NE(string_storage, nullptr);
    auto string_handle = segment.get_handle_from_address(string_storage);
    gqe::storage::shared_string_compressed_sliced_column<false> string_column(segment,
                                                                              string_handle);

    EXPECT_EQ(string_column.size(), 4);
    EXPECT_FALSE(string_column.is_large_string());
    auto string_char_partition   = string_column.char_buffer_view().get_partition(0);
    auto string_offset_partition = string_column.offset_buffer_view().get_partition(0);
    EXPECT_GT(string_char_partition.size(), 0);
    EXPECT_TRUE(is_in_segment(segment, string_char_partition.data()));
    EXPECT_GT(string_offset_partition.size(), 0);
    EXPECT_TRUE(is_in_segment(segment, string_offset_partition.data()));
    EXPECT_GT(string_column.get_char_partition_output_size(0), 0);

    auto const free_memory_before_teardown = segment.get_free_memory();

    string_column.deallocate(stream, shared_mr_ref);
    EXPECT_EQ(
      segment
        .find<gqe::storage::shared_string_compressed_sliced_column_storage<false>>("string_column")
        .first,
      nullptr);
    EXPECT_NO_THROW(string_column.deallocate(stream, shared_mr_ref));

    fixed_width_interface.deallocate(stream, shared_mr_ref);
    EXPECT_EQ(
      segment.find<gqe::storage::shared_compressed_sliced_column_storage>("fixed_width_column")
        .first,
      nullptr);
    EXPECT_NO_THROW(fixed_width_interface.deallocate(stream, shared_mr_ref));
    EXPECT_THROW(fixed_width_interface.size(), std::logic_error);

    EXPECT_GT(segment.get_free_memory(), free_memory_before_teardown);
  }
  boost::interprocess::shared_memory_object::remove(segment_name);
}

TEST(CompressedSlicedColumnLocalTest, ExplicitDeallocateFreesFixedWidthAndStringBuffers)
{
  auto stream = cudf::get_default_stream();
  rmm::mr::cuda_memory_resource mr;
  auto kind = gqe::memory_kind::device{rmm::cuda_device_id(0)};

  cudf::test::fixed_width_column_wrapper<int32_t> fixed_width_values({1, 2, 3, 4});
  gqe::storage::compressed_sliced_column fixed_width_column(
    cudf::column(fixed_width_values),
    2,
    kind,
    make_shared_storage_compression_config(),
    stream,
    mr);
  gqe::storage::compressed_sliced_column_interface& fixed_width_interface = fixed_width_column;
  EXPECT_GT(fixed_width_interface.data_buffer_view().get_partition(0).size(), 0);
  EXPECT_NO_THROW(fixed_width_interface.deallocate(stream, mr));

  cudf::test::strings_column_wrapper string_values({"aa", "bbb", "c", "dddd"});
  gqe::storage::string_compressed_sliced_column<false> string_column(
    cudf::column(string_values), 2, kind, make_shared_storage_compression_config(), stream, mr);
  EXPECT_GT(string_column.char_buffer_view().get_partition(0).size(), 0);
  EXPECT_GT(string_column.offset_buffer_view().get_partition(0).size(), 0);
  EXPECT_NO_THROW(string_column.deallocate(stream, mr));
}

TEST(ContiguousColumnLocalTest, ConstructsFixedWidthAndStringStorageFromCudfViews)
{
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  cudf::test::fixed_width_column_wrapper<int32_t> fixed_width_values({1, 2, 3, 4},
                                                                     {true, false, true, true});
  auto fixed_width_source = cudf::column(fixed_width_values);
  gqe::storage::contiguous_column fixed_width_column(
    cudf::column(fixed_width_source.view()), stream, mr);
  gqe::storage::contiguous_column_interface const& fixed_width_interface = fixed_width_column;

  EXPECT_EQ(fixed_width_interface.layout(), gqe::storage::column_layout::contiguous);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_interface.view(), fixed_width_source.view());
  EXPECT_EQ(fixed_width_column.mutable_view().size(), fixed_width_source.size());
  EXPECT_EQ(fixed_width_column.get_uncompressed_size(),
            fixed_width_source.size() * cudf::size_of(fixed_width_source.type()));

  cudf::test::strings_column_wrapper string_values({"aa", "bbb", "c", "dddd"});
  auto string_source = cudf::column(string_values);
  gqe::storage::contiguous_column string_column(cudf::column(string_source.view()), stream, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(string_column.view(), string_source.view());
  EXPECT_EQ(string_column.mutable_view().num_children(), string_source.num_children());
  EXPECT_GT(string_column.get_uncompressed_size(), 0);

  EXPECT_NO_THROW(fixed_width_column.deallocate(stream, mr));
  EXPECT_NO_THROW(string_column.deallocate(stream, mr));
}

TEST(ContiguousColumnSharedMemoryTest, SegmentConstructsFixedWidthAndStringColumns)
{
  static_assert(!std::is_polymorphic_v<gqe::storage::shared_contiguous_column_storage>);

  constexpr char segment_name[] = "gqe_contiguous_column_allocator_test";
  boost::interprocess::shared_memory_object::remove(segment_name);
  {
    [[maybe_unused]] boost::interprocess::managed_shared_memory created_segment(
      boost::interprocess::create_only, segment_name, 8 << 20);

    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only,
                                                       segment_name);
    auto shared_mr = std::make_shared<gqe::memory_resource::boost_shared_memory_resource>(&segment);
    auto shared_mr_ref = gqe::memory_resource::shared_resource_adaptor{shared_mr};
    gqe::shared_memory_allocator alloc(segment.get_segment_manager());
    auto stream = cudf::get_default_stream();

    cudf::test::fixed_width_column_wrapper<int32_t> fixed_width_values({1, 2, 3, 4},
                                                                       {true, true, false, true});
    auto fixed_width_source   = cudf::column(fixed_width_values);
    auto* fixed_width_storage = segment.construct<gqe::storage::shared_contiguous_column_storage>(
      "fixed_width_column")(fixed_width_source.view(), stream, shared_mr_ref, alloc);
    ASSERT_NE(fixed_width_storage, nullptr);
    gqe::storage::shared_contiguous_column fixed_width_column(
      segment, segment.get_handle_from_address(fixed_width_storage));
    gqe::storage::contiguous_column_interface const& fixed_width_interface = fixed_width_column;

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_interface.view(), fixed_width_source.view());
    EXPECT_TRUE(is_in_segment(segment, fixed_width_column.view().head()));
    EXPECT_EQ(fixed_width_column.null_count(), fixed_width_source.null_count());

    cudf::test::strings_column_wrapper string_values({"aa", "bbb", "c", "dddd"});
    auto string_source   = cudf::column(string_values);
    auto* string_storage = segment.construct<gqe::storage::shared_contiguous_column_storage>(
      "string_column")(string_source.view(), stream, shared_mr_ref, alloc);
    ASSERT_NE(string_storage, nullptr);
    gqe::storage::shared_contiguous_column string_column(
      segment, segment.get_handle_from_address(string_storage));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(string_column.view(), string_source.view());
    EXPECT_TRUE(is_in_segment(segment, string_column.view().head()));
    EXPECT_TRUE(is_in_segment(segment, string_column.view().child(0).head()));
    EXPECT_GT(string_column.get_uncompressed_size(), 0);

    auto const free_memory_before_teardown = segment.get_free_memory();

    string_column.deallocate(stream, shared_mr_ref);
    EXPECT_EQ(segment.find<gqe::storage::shared_contiguous_column_storage>("string_column").first,
              nullptr);
    EXPECT_NO_THROW(string_column.deallocate(stream, shared_mr_ref));

    fixed_width_column.deallocate(stream, shared_mr_ref);
    EXPECT_EQ(
      segment.find<gqe::storage::shared_contiguous_column_storage>("fixed_width_column").first,
      nullptr);
    EXPECT_NO_THROW(fixed_width_column.deallocate(stream, shared_mr_ref));

    EXPECT_GT(segment.get_free_memory(), free_memory_before_teardown);
  }
  boost::interprocess::shared_memory_object::remove(segment_name);
}

class SharedInMemoryTableTest : public gqe::test::BaseFixture {};

// Exercises the statistics-table write primitive on shared_in_memory_table: replace_contents()
// publishes a single row group and frees the previously published one, so repeated replaces keep
// the table at one row group (replace, not append) and do not leak segment memory;
// clear_row_groups() empties the table. Each mutation advances storage_revision. Counts are read
// through a fresh facade because the writing facade drops its local wrappers on replace (real
// readers reattach from shared metadata).
TEST_F(SharedInMemoryTableTest, ReplaceContentsReplacesAndReclaimsThenClears)
{
  constexpr char segment_name[] = "gqe_shared_table_replace_test";
  boost::interprocess::shared_memory_object::remove(segment_name);
  {
    boost::interprocess::managed_shared_memory segment(
      boost::interprocess::create_only, segment_name, 32 << 20);
    auto shared_mr = std::make_shared<gqe::memory_resource::boost_shared_memory_resource>(&segment);
    auto shared_mr_ref = gqe::memory_resource::shared_resource_adaptor{shared_mr};
    gqe::shared_memory_allocator alloc(segment.get_segment_manager());
    auto stream = cudf::get_default_stream();

    auto* table_storage = segment.construct<gqe::storage::shared_in_memory_table_storage>(
      boost::interprocess::anonymous_instance)(alloc);
    auto const table_handle = segment.get_handle_from_address(table_storage);
    std::vector<std::string> const col_names{"a", "b"};
    std::vector<cudf::data_type> const col_types{cudf::data_type{cudf::type_id::INT32},
                                                 cudf::data_type{cudf::type_id::INT32}};

    // A fresh facade observes the currently published row groups (reads shared metadata on
    // construction), unlike the writing facade which drops its wrappers on replace.
    auto num_row_groups = [&] {
      return gqe::storage::shared_in_memory_table(&segment, col_names, col_types, table_handle)
        .num_row_groups();
    };
    auto revision = [&] {
      return gqe::storage::shared_in_memory_table(&segment, col_names, col_types, table_handle)
        .storage_revision();
    };

    gqe::storage::shared_in_memory_table writer(&segment, col_names, col_types, table_handle);
    EXPECT_EQ(num_row_groups(), 0u);
    auto const revision_empty = revision();

    cudf::test::fixed_width_column_wrapper<int32_t> a0({1, 2, 3});
    cudf::test::fixed_width_column_wrapper<int32_t> b0({4, 5, 6});
    std::vector<cudf::column_view> const input0{a0, b0};
    writer.replace_contents(cudf::table_view{input0}, shared_mr_ref, stream);
    EXPECT_EQ(num_row_groups(), 1u);
    EXPECT_GT(revision(), revision_empty);

    // Repeated same-size replaces must keep exactly one row group (replace, not append) and must
    // not shrink free memory (the previous row group is freed each time).
    auto const free_after_first = segment.get_free_memory();
    for (int i = 0; i < 5; ++i) {
      cudf::test::fixed_width_column_wrapper<int32_t> a({7, 8, 9});
      cudf::test::fixed_width_column_wrapper<int32_t> b({10, 11, 12});
      std::vector<cudf::column_view> const input{a, b};
      writer.replace_contents(cudf::table_view{input}, shared_mr_ref, stream);
      EXPECT_EQ(num_row_groups(), 1u);
    }
    EXPECT_GE(segment.get_free_memory(), free_after_first);

    auto const revision_before_clear = revision();
    writer.clear_row_groups(stream, shared_mr_ref);
    EXPECT_EQ(num_row_groups(), 0u);
    EXPECT_GT(revision(), revision_before_clear);
  }
  boost::interprocess::shared_memory_object::remove(segment_name);
}

TEST_F(SharedInMemoryTableTest, ReplacementSurvivesDetachedCleanupFailure)
{
  constexpr char segment_name[] = "gqe_shared_table_cleanup_failure_test";
  boost::interprocess::shared_memory_object::remove(segment_name);
  {
    boost::interprocess::managed_shared_memory segment(
      boost::interprocess::create_only, segment_name, 32 << 20);
    auto shared_mr = std::make_shared<gqe::memory_resource::boost_shared_memory_resource>(&segment);
    auto shared_mr_ref = gqe::memory_resource::shared_resource_adaptor{shared_mr};
    gqe::shared_memory_allocator alloc(segment.get_segment_manager());
    auto stream = cudf::get_default_stream();

    auto* table_storage = segment.construct<gqe::storage::shared_in_memory_table_storage>(
      boost::interprocess::anonymous_instance)(alloc);
    auto const table_handle = segment.get_handle_from_address(table_storage);
    std::vector<std::string> const col_names{"a"};
    std::vector<cudf::data_type> const col_types{cudf::data_type{cudf::type_id::INT32}};

    // Construct the writer before injecting malformed old metadata so it does not try to attach to
    // that row group itself.
    gqe::storage::shared_in_memory_table writer(&segment, col_names, col_types, table_handle);

    cudf::test::fixed_width_column_wrapper<int32_t> old_values({1, 2, 3});
    auto old_source          = cudf::column(old_values);
    auto* old_column_storage = segment.construct<gqe::storage::shared_contiguous_column_storage>(
      boost::interprocess::anonymous_instance)(old_source.view(), stream, shared_mr_ref, alloc);
    auto const old_column_handle = segment.get_handle_from_address(old_column_storage);
    stream.synchronize();

    gqe::storage::shared_row_group_storage old_row_group(old_source.size(), alloc);
    old_row_group.columns.emplace_back(static_cast<gqe::storage::column_layout>(-1),
                                       old_column_handle);
    auto const old_row_group_handle    = table_storage->append(segment, std::move(old_row_group));
    auto const revision_before_replace = writer.storage_revision();

    cudf::test::fixed_width_column_wrapper<int32_t> replacement_values({7, 8, 9});
    auto replacement_source = cudf::column(replacement_values);
    std::vector<cudf::column_view> const replacement_input{replacement_source.view()};

    // Cleanup reconstructs the detached old row group, sees the invalid layout, and throws
    // internally. The exception must be contained because the replacement is already published.
    EXPECT_NO_THROW(
      writer.replace_contents(cudf::table_view{replacement_input}, shared_mr_ref, stream));
    EXPECT_GT(writer.storage_revision(), revision_before_replace);

    {
      gqe::storage::shared_in_memory_table fresh_reader(
        &segment, col_names, col_types, table_handle);
      ASSERT_EQ(fresh_reader.num_row_groups(), 1u);
      ASSERT_NE(fresh_reader.row_group_at(0), nullptr);
      auto const* replacement_column =
        dynamic_cast<gqe::storage::contiguous_column_interface const*>(
          &fresh_reader.row_group_at(0)->get_column(0));
      ASSERT_NE(replacement_column, nullptr);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(replacement_column->view(), replacement_source.view());
    }

    // The injected cleanup failure intentionally left the detached old storage allocated. Restore
    // its layout and reclaim it explicitly so the test itself does not leak shared storage.
    auto* detached_storage = static_cast<gqe::storage::shared_row_group_storage*>(
      segment.get_address_from_handle(old_row_group_handle));
    detached_storage->columns.front().layout = gqe::storage::column_layout::contiguous;
    gqe::storage::shared_row_group detached_old(segment, old_row_group_handle);
    detached_old.deallocate(stream, shared_mr_ref);

    writer.clear_row_groups(stream, shared_mr_ref);
  }
  boost::interprocess::shared_memory_object::remove(segment_name);
}

TEST_F(SharedInMemoryTableTest, UnregisterBoostSharedTableReclaimsSharedStorage)
{
  boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
  {
    [[maybe_unused]] boost::interprocess::managed_shared_memory created_segment(
      boost::interprocess::create_only, gqe::node_manager::shared_memory_name, 8 << 20);

    auto& segment                          = get_task_manager_ctx()->shared_memory_segment();
    auto const free_memory_before_register = segment.get_free_memory();

    gqe::catalog catalog{get_task_manager_ctx()};
    catalog.register_table("shared_table",
                           {{"a", cudf::data_type{cudf::type_id::INT32}}},
                           gqe::storage_kind::boost_shared_memory{},
                           gqe::partitioning_schema_kind::none{});
    auto const free_memory_after_register = segment.get_free_memory();
    EXPECT_LT(free_memory_after_register, free_memory_before_register);

    auto shared_table = std::dynamic_pointer_cast<gqe::storage::shared_in_memory_table>(
      catalog.table_storage("shared_table"));
    ASSERT_NE(shared_table, nullptr);
    auto stream = cudf::get_default_stream();
    auto shared_mr =
      get_task_manager_ctx()->get_table_memory_resource(gqe::memory_kind::boost_shared{});
    gqe::shared_memory_allocator alloc(segment.get_segment_manager());
    cudf::test::fixed_width_column_wrapper<int32_t> values({1, 2, 3, 4});
    auto source          = cudf::column(values);
    auto* column_storage = segment.construct<gqe::storage::shared_contiguous_column_storage>(
      boost::interprocess::anonymous_instance)(source.view(), stream, shared_mr, alloc);
    auto const column_handle = segment.get_handle_from_address(column_storage);
    gqe::storage::shared_row_group_storage row_group(source.size(), alloc);
    row_group.columns.push_back({gqe::storage::column_layout::contiguous, column_handle});
    shared_table->get_row_group_appender()(std::move(row_group));
    stream.synchronize();
    EXPECT_EQ(shared_table->num_row_groups(), 0);

    catalog.unregister_table("shared_table");
    EXPECT_FALSE(catalog.has_table("shared_table"));
    EXPECT_EQ(segment.get_free_memory(), free_memory_before_register);
  }
  boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
}

TEST_F(SharedInMemoryTableTest, CatalogIdentityAndRevisionTrackStorageChanges)
{
  boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
  {
    [[maybe_unused]] boost::interprocess::managed_shared_memory created_segment(
      boost::interprocess::create_only, gqe::node_manager::shared_memory_name, 8 << 20);

    gqe::catalog catalog{get_task_manager_ctx()};
    auto register_table = [&] {
      catalog.register_table("shared_table",
                             {{"a", cudf::data_type{cudf::type_id::INT32}}},
                             gqe::storage_kind::boost_shared_memory{},
                             gqe::partitioning_schema_kind::none{});
    };

    register_table();
    auto const first_descriptor = std::get<gqe::storage::boost_shared_memory_descriptor>(
      catalog.storage_descriptor("shared_table"));
    auto const repeated_descriptor = std::get<gqe::storage::boost_shared_memory_descriptor>(
      catalog.storage_descriptor("shared_table"));
    EXPECT_EQ(first_descriptor.table_id, repeated_descriptor.table_id);
    EXPECT_EQ(first_descriptor.storage_revision, 0);
    EXPECT_EQ(repeated_descriptor.storage_revision, 0);

    auto shared_table = std::dynamic_pointer_cast<gqe::storage::shared_in_memory_table>(
      catalog.table_storage("shared_table"));
    ASSERT_NE(shared_table, nullptr);
    auto& segment = get_task_manager_ctx()->shared_memory_segment();
    gqe::shared_memory_allocator alloc(segment.get_segment_manager());
    shared_table->get_row_group_appender()(gqe::storage::shared_row_group_storage(0, alloc));
    EXPECT_EQ(shared_table->num_row_groups(), 0);

    auto const appended_descriptor = std::get<gqe::storage::boost_shared_memory_descriptor>(
      catalog.storage_descriptor("shared_table"));
    EXPECT_EQ(appended_descriptor.table_id, first_descriptor.table_id);
    EXPECT_EQ(appended_descriptor.storage_revision, 1);

    catalog.unregister_table("shared_table");
    register_table();
    auto const recreated_descriptor = std::get<gqe::storage::boost_shared_memory_descriptor>(
      catalog.storage_descriptor("shared_table"));
    EXPECT_NE(first_descriptor.table_id, recreated_descriptor.table_id);
    EXPECT_EQ(recreated_descriptor.storage_revision, 0);
    catalog.unregister_table("shared_table");
  }
  boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
}

TEST_F(SharedInMemoryTableTest, AttachFromHandleAndRead)
{
  boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
  {
    [[maybe_unused]] boost::interprocess::managed_shared_memory created_segment(
      boost::interprocess::create_only, gqe::node_manager::shared_memory_name, 8 << 20);

    cudf::test::fixed_width_column_wrapper<int32_t> values({1, 2, 3, 4});
    auto source = cudf::column(values);
    std::shared_ptr<boost::interprocess::managed_shared_memory> shared_segment;
    std::shared_ptr<gqe::memory_resource::boost_shared_memory_resource> shared_mr;
    gqe::shared_memory_handle_t table_handle{};
    {
      shared_segment = std::make_shared<boost::interprocess::managed_shared_memory>(
        boost::interprocess::open_only, gqe::node_manager::shared_memory_name);
      shared_mr =
        std::make_shared<gqe::memory_resource::boost_shared_memory_resource>(shared_segment.get());
      auto shared_mr_ref = gqe::memory_resource::shared_resource_adaptor{shared_mr};
      auto& segment      = *shared_segment;
      gqe::shared_memory_allocator alloc(segment.get_segment_manager());
      auto stream = cudf::get_default_stream();

      auto* table_storage =
        segment.construct<gqe::storage::shared_in_memory_table_storage>("table")(alloc);
      ASSERT_NE(table_storage, nullptr);
      table_handle = segment.get_handle_from_address(table_storage);
      gqe::storage::shared_in_memory_table owner(
        &segment, {"a"}, {cudf::data_type{cudf::type_id::INT32}}, table_handle);
      table_handle = owner.shared_table_handle();
      ASSERT_NE(table_handle, gqe::shared_memory_handle_t{});

      auto* column_storage = segment.construct<gqe::storage::shared_contiguous_column_storage>(
        "column")(source.view(), stream, shared_mr_ref, alloc);
      ASSERT_NE(column_storage, nullptr);
      auto const column_handle = segment.get_handle_from_address(column_storage);

      gqe::storage::shared_row_group_storage row_group(source.size(), alloc);
      row_group.columns.push_back({gqe::storage::column_layout::contiguous, column_handle});
      owner.get_row_group_appender()(std::move(row_group));
      stream.synchronize();
      ASSERT_EQ(owner.num_row_groups(), 0);
    }

    gqe::storage::shared_in_memory_table table(
      shared_segment.get(), {"a"}, {cudf::data_type{cudf::type_id::INT32}}, table_handle);
    auto readable = table.readable_view();

    std::vector<gqe::storage::readable_view::task_parameters> task_parameters;
    task_parameters.emplace_back(gqe::storage::readable_view::task_parameters{0});
    gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
    auto tasks = readable->get_read_tasks(
      std::move(task_parameters), ctx_ref, 0, {"a"}, {cudf::data_type{cudf::type_id::INT32}});
    ASSERT_EQ(tasks.size(), 1);
    tasks.front()->execute();
    auto result = tasks.front()->result();

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->num_columns(), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->column(0), source.view());

    tasks.clear();
    readable.reset();
    auto shared_mr_ref = gqe::memory_resource::shared_resource_adaptor{shared_mr};
    table.deallocate(cudf::get_default_stream(), shared_mr_ref);
  }
  boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
}

// Test fixture for in_memory_read_task::execute
//
// The fixture creates four input tables. Each table has four columns, two integer columns, and two
// double columns. In the second column of the integer and double columns, every other element is a
// null value. The four tables have 100 rows each and the values over all columns go from 0 to 399.
//
// There are convenience methods to create row groups from one or more input tables, to create a
// range filter that can be used as a partial filter, to create comparison tables, and to create a
// read_task to test.
//
// Tests can change parameters of the read task by manipulating _task_manager_ctx and _query_ctx
// before creating the read task, e.g., to compress the row group columns.
static constexpr cudf::size_type DEFAULT_NUM_ROWS       = 100;
static constexpr cudf::size_type DEFAULT_PARTITION_SIZE = 5;
class InMemoryReadTaskTest : public gqe::test::BaseFixture {
 protected:
  InMemoryReadTaskTest()
    : _query_ctx(get_query_ctx()),
      _ctx_ref(gqe::context_reference{get_task_manager_ctx(), _query_ctx}),
      _task_id(0),
      _stage_id(0),
      _memory_kind(gqe::memory_kind::device{rmm::cuda_device_id(0)}),
      _memory_resource(rmm::mr::cuda_memory_resource{}),
      _num_rows(DEFAULT_NUM_ROWS),
      _partition_size(DEFAULT_PARTITION_SIZE)
  {
    rmm::mr::set_current_device_resource(_memory_resource);
  }

  void SetUp(const cudf::size_type num_rows       = DEFAULT_NUM_ROWS,
             const cudf::size_type partition_size = DEFAULT_PARTITION_SIZE,
             const bool use_string_column         = false,
             const size_t num_tables              = NUM_TABLES,
             const bool use_host_allocator        = false)
  {
    _num_rows       = num_rows;
    _partition_size = partition_size;

    // Explicitly enable pruning, in case it is disabled by default.
    _query_ctx->parameters.use_partition_pruning   = true;
    _query_ctx->parameters.zone_map_partition_size = partition_size;

    if (use_host_allocator) {
      _memory_resource = cuda::mr::any_resource<cuda::mr::device_accessible>{
        gqe::memory_resource::pinned_memory_resource{}};
      rmm::mr::set_current_device_resource(_memory_resource);
    }

    // Create the four input tables, the corresponding column indexes, and the data types.
    for (size_t i = 0; i < num_tables; ++i) {
      auto table = create_input_table(_num_rows, i * _num_rows, use_string_column);
      _input_tables.push_back(table->view());
      _tables.push_back(std::move(table));
    }
    _column_indexes = {0, 1, 2, 3};
    _data_types     = {cudf::data_type(cudf::type_id::INT32),
                       cudf::data_type(cudf::type_id::INT32),
                       cudf::data_type(cudf::type_id::FLOAT32),
                       cudf::data_type(cudf::type_id::FLOAT32)};
    if (use_string_column) {
      _column_indexes.push_back(4);
      _data_types.push_back(cudf::data_type(cudf::type_id::STRING));
    }
  }

  // Create a typed input column with optional null values
  template <typename T>
  cudf::test::fixed_width_column_wrapper<T> create_fixed_width_column_wrapper(
    cudf::size_type num_rows, cudf::size_type offset, bool odds_are_null = false)
  {
    std::vector<T> values(num_rows);
    std::iota(values.begin(), values.end(), static_cast<T>(offset));
    // TODO Support null values
    if (false && odds_are_null) {
      std::vector<bool> is_null(num_rows);
      std::transform(values.begin(), values.end(), is_null.begin(), [](auto i) {
        return static_cast<int32_t>(i) % 2 == 0;
      });
      cudf::test::fixed_width_column_wrapper<T> col(values.begin(), values.end(), is_null.begin());
      return col;
    } else {
      cudf::test::fixed_width_column_wrapper<T> col(values.begin(), values.end());
      return col;
    }
  }

  // Create a string column with null values, strings a rotating length between 1 and 100
  std::unique_ptr<cudf::column> create_string_column(cudf::size_type num_rows,
                                                     cudf::size_type offset,
                                                     cudf::size_type char_offset)
  {
    std::vector<int32_t> offsets(num_rows + 1);
    int32_t offset_val = 0;
    for (cudf::size_type i = 0; i < num_rows; i++) {
      int32_t string_length = (i + offset) % 4 + 1;
      offsets[i]            = offset_val;
      offset_val += string_length;
    }

    offsets[num_rows] = offset_val;

    std::vector<char> values(offset_val);
    for (cudf::size_type i = 0; i < offset_val; i++) {
      values[i] = 'a' + ((i + char_offset) % 26);
    }

    rmm::device_buffer offsets_buffer(reinterpret_cast<uint8_t*>(offsets.data()),
                                      (num_rows + 1) * sizeof(int32_t),
                                      cudf::get_default_stream(),
                                      _memory_resource);
    rmm::device_buffer values_buffer(reinterpret_cast<uint8_t*>(values.data()),
                                     offset_val,
                                     cudf::get_default_stream(),
                                     _memory_resource);
    std::unique_ptr<cudf::column> offsets_col =
      std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                     num_rows + 1,
                                     std::move(offsets_buffer),
                                     rmm::device_buffer(),
                                     0);
    return cudf::make_strings_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(values_buffer),
                                     0 /* null count */,
                                     rmm::device_buffer{});
  }

  // Create a cudf::table consisting of 2 integer columns (one without nulls and one with nulls) and
  // 2 float columns (same).
  std::unique_ptr<cudf::table> create_input_table(cudf::size_type num_rows,
                                                  cudf::size_type offset,
                                                  const bool add_string_column = false,
                                                  cudf::size_type char_offset  = 0)
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(create_fixed_width_column_wrapper<int32_t>(num_rows, offset).release());
    columns.push_back(create_fixed_width_column_wrapper<int32_t>(num_rows, offset, true).release());
    columns.push_back(create_fixed_width_column_wrapper<float>(num_rows, offset).release());
    columns.push_back(create_fixed_width_column_wrapper<float>(num_rows, offset, true).release());
    if (add_string_column) {
      columns.push_back(create_string_column(num_rows, offset, char_offset));
    }
    return std::make_unique<cudf::table>(
      std::move(columns), cudf::get_default_stream(), _memory_resource);
  }

  // Create a range filter start <= column < end
  template <typename T>
  std::unique_ptr<gqe::expression> create_range_filter(cudf::size_type column_index,
                                                       T start_inclusive,
                                                       T end_exclusive)
  {
    return std::make_unique<gqe::logical_and_expression>(
      std::make_unique<gqe::less_equal_expression>(
        std::make_unique<gqe::literal_expression<T>>(start_inclusive),
        std::make_unique<gqe::column_reference_expression>(column_index)),
      std::make_unique<gqe::less_expression>(
        std::make_unique<gqe::column_reference_expression>(column_index),
        std::make_unique<gqe::literal_expression<T>>(end_exclusive)));
  }

  template <typename T>
  std::unique_ptr<gqe::expression> create_two_range_filters(cudf::size_type column_index,
                                                            T start_inclusive1,
                                                            T end_exclusive1,
                                                            T start_inclusive2,
                                                            T end_exclusive2)
  {
    return std::make_unique<gqe::logical_or_expression>(
      create_range_filter(column_index, start_inclusive1, end_exclusive1),
      create_range_filter(column_index, start_inclusive2, end_exclusive2));
  }

  // Create row groups from input table_views
  std::vector<const gqe::storage::row_group_interface*> create_row_groups(
    std::vector<cudf::table_view>::iterator begin, std::vector<cudf::table_view>::iterator end)
  {
    std::vector<const gqe::storage::row_group_interface*> result;
    std::transform(
      begin, end, std::back_inserter(result), [&](const cudf::table_view& input_table) {
        auto const comp_format                 = _query_ctx->parameters.compression_format;
        auto const chunk_size                  = _query_ctx->parameters.compression_chunk_size;
        auto const compression_ratio_threshold = _query_ctx->parameters.compression_ratio_threshold;
        auto const secondary_compression_format =
          _query_ctx->parameters.secondary_compression_format;
        auto const secondary_compression_ratio_threshold =
          _query_ctx->parameters.secondary_compression_ratio_threshold;
        auto const secondary_compression_multiplier_threshold =
          _query_ctx->parameters.secondary_compression_multiplier_threshold;
        auto const use_cpu_compression = _query_ctx->parameters.use_cpu_compression;
        auto const compression_level   = _query_ctx->parameters.compression_level;
        auto const decompress_backend  = _query_ctx->parameters.decompression_backend;
        std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
        std::transform(
          input_table.begin(),
          input_table.end(),
          std::back_inserter(columns),
          [comp_format,
           chunk_size,
           compression_ratio_threshold,
           secondary_compression_format,
           secondary_compression_ratio_threshold,
           secondary_compression_multiplier_threshold,
           use_cpu_compression,
           compression_level,
           decompress_backend,
           this](
            const cudf::column_view& column_view) -> std::unique_ptr<gqe::storage::column_base> {
            if (comp_format == gqe::compression_format::none) {
              return std::make_unique<gqe::storage::contiguous_column>(cudf::column(column_view));
            }
            auto compression_config = gqe::compression::configuration{
              .primary_compression_format            = comp_format,
              .secondary_compression_format          = secondary_compression_format,
              .compression_chunk_size                = chunk_size,
              .compression_ratio_threshold           = compression_ratio_threshold,
              .secondary_compression_ratio_threshold = secondary_compression_ratio_threshold,
              .secondary_compression_multiplier_threshold =
                secondary_compression_multiplier_threshold,
              .use_cpu_compression = use_cpu_compression,
              .compression_level   = compression_level,
              .decompress_backend  = decompress_backend};
            if (column_view.type().id() == cudf::type_id::STRING) {
              return std::make_unique<gqe::storage::string_compressed_sliced_column<false>>(
                cudf::column(column_view),
                _partition_size,
                _memory_kind,
                std::move(compression_config),
                cudf::get_default_stream(),
                _memory_resource);
            } else {
              return std::make_unique<gqe::storage::compressed_sliced_column>(
                cudf::column(column_view),
                _partition_size,
                _memory_kind,
                std::move(compression_config),
                cudf::get_default_stream(),
                _memory_resource);
            }
          });

        auto zone_map = std::make_unique<gqe::local_zone_map>(input_table, _partition_size);
        auto row_group =
          std::make_unique<gqe::storage::row_group>(std::move(columns), std::move(zone_map));
        auto row_group_ptr = row_group.get();
        _row_groups.push_back(std::move(row_group));
        // The pointer cannot escape the function because it is stored in _row_groups
        return row_group_ptr;
      });
    return result;
  }

  // Convenience method to create an in_memory_read_task, so that the tests don't have to pass
  // parameters that do not affect the test result, e.g., the task_id.
  gqe::storage::in_memory_read_task create_read_task(
    std::vector<const gqe::storage::row_group_interface*> row_groups,
    std::vector<cudf::size_type> column_indexes,
    std::vector<cudf::data_type> data_types,
    std::optional<arrow::compute::Expression> zone_map_filter = std::nullopt)
  {
    return gqe::storage::in_memory_read_task(_ctx_ref,
                                             _task_id,
                                             _stage_id,
                                             row_groups,
                                             column_indexes,
                                             data_types,
                                             _memory_kind,
                                             zone_map_filter);
  }

  // Number of input tables
  static constexpr size_t NUM_TABLES = 4;

  // Parameters that are required by the constructor of in_memory_read_task but don't really change
  // across the tests. The only exception is _query_ctx.parameters, which can be manipulated to
  // enable/disable zero-copy reads and compression.
  gqe::query_context* _query_ctx;
  const gqe::context_reference _ctx_ref;
  const int32_t _task_id;
  const int32_t _stage_id;
  const gqe::memory_kind::type _memory_kind;

  // The input tables need be accessible on the GPU, otherwise cudf::concatenate does not process
  // null counts correctly. The constructor sets this to CUDA device memory.
  cuda::mr::any_resource<cuda::mr::device_accessible> _memory_resource;

  // Parameters that influence the input (i.e., input table views from which row groups are
  // constructed), and the result of in_memory_read_task::execute (i.e., the requested columns and
  // datatypes)
  std::vector<cudf::table_view> _input_tables;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;

  // Input tables created in the constructor need to be kept valid during the lifetime of the test
  std::vector<std::unique_ptr<cudf::table>> _tables;

  // Row groups that are created by create_row_groups need to be kept valid during the lifetime of
  // the test
  std::vector<std::unique_ptr<gqe::storage::row_group>> _row_groups;

  // Number of rows per input table
  cudf::size_type _num_rows;

  // Number of rows per zone map partition
  cudf::size_type _partition_size;
};

TEST_F(InMemoryReadTaskTest, returnSingleRowGroupDirectly)
{
  SetUp();
  // Create a task with a single (!) row group
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  // The result contains the contents of the row group and is passed as a borrowed result
  ASSERT_TRUE(task.result().has_value());
  EXPECT_FALSE(*task.is_result_owned());
  cudf::table_view expected = _input_tables[0];
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected);
}

TEST_F(InMemoryReadTaskTest, copySingleRowGroupIfZeroCopyIsDisabled)
{
  SetUp();
  // Disable zero-copying
  _query_ctx->parameters.read_use_zero_copy = false;
  // Create a task with a single row group
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  // The result contains the contents of the row group and is passed as an owned result
  ASSERT_TRUE(task.result().has_value());
  EXPECT_TRUE(*task.is_result_owned());
  cudf::table_view expected = _input_tables[0];
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected);
}

TEST_F(InMemoryReadTaskTest, CopyContiguousStringColumnWhenZeroCopyDisabled)
{
  SetUp(DEFAULT_NUM_ROWS, DEFAULT_PARTITION_SIZE, true /* use_string_column */);
  _query_ctx->parameters.read_use_zero_copy = false;

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  task.execute();

  ASSERT_TRUE(task.result().has_value());
  EXPECT_TRUE(*task.is_result_owned());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), _input_tables[0]);
}

TEST_F(InMemoryReadTaskTest, concatenateMultipleRowGroups)
{
  SetUp();
  // Create a task with multiple row groups
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  // The result contains the contents of the row group and is passed as a borrowed result
  ASSERT_TRUE(task.result().has_value());
  EXPECT_TRUE(*task.is_result_owned());
  auto expected = cudf::concatenate(_input_tables);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, returnEmptyTableIfEntireInputIsPruned)
{
  SetUp();
  // Create a zone map filter that exclude all inputs. If there are N input tables, column_0 > N *
  // _num_rows should filter out all rows.
  std::unique_ptr<gqe::expression> partial_filter = std::make_unique<gqe::greater_expression>(
    std::make_unique<gqe::column_reference_expression>(0),
    std::make_unique<gqe::literal_expression<int32_t>>(_input_tables.size() * _num_rows));
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with row groups (actual number should not matter), and pass the filter
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result is empty
  ASSERT_TRUE(task.result().has_value());
  EXPECT_EQ(task.result()->num_rows(), 0);
}

TEST_F(InMemoryReadTaskTest, returnSingleRowGroupAfterPruningDirectly)
{
  SetUp();
  // Create a zone map filter 148 <= col0 < 157. This should return three partitions, from 145
  // (inclusive) to 160 (exclusive), i.e., 15 rows in total. The rows are contained in the second
  // input table row group.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter<int32_t>(0, 148, 157);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as a borrowed
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_FALSE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result     = 15;
  constexpr cudf::size_type start_offset_in_result = 145;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, concatenatePartitionsInMultipleRowGroups)
{
  SetUp();
  // Create a zone map filter 198 <= col0 < 207. This should return three partitions, from 195
  // (inclusive) to 210 (exclusive), i.e., 15 rows in total. The first partition is contained in the
  // second row group and the other partitions in the third row group.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter<int32_t>(0, 198, 207);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple row groups, and pass the filter. The filter will prune all but
  // two consecutive row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the two row groups and is passed as an
  // owned result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result     = 15;
  constexpr cudf::size_type start_offset_in_result = 195;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, concatenateDiscontiguousPartitionsInSingleRowGroup)
{
  SetUp();
  // Create a zone map filter 118 <= col0 < 127 OR 168.0 <= col2 <= 177.0 (the different columns
  // don't really matter). This should return six partitions: three from 115 (inclusive) to 130
  // (exclusive) and three from 165 (inclusive) to 180 (exclusive). Both are contained in the second
  // row group.
  std::unique_ptr<gqe::expression> partial_filter = std::make_unique<gqe::logical_or_expression>(
    create_range_filter<int32_t>(0, 118, 127), create_range_filter<double>(2, 165, 180));
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple row groups, and pass the filter. The filter will prune all but
  // one row group.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying discontiguous partitions of the single row group and is
  // passed as an owned result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result       = 15;
  constexpr cudf::size_type start_offset_in_result_1 = 115;
  constexpr cudf::size_type start_offset_in_result_2 = 165;
  auto expected1 = create_input_table(num_rows_in_result, start_offset_in_result_1);
  auto expected2 = create_input_table(num_rows_in_result, start_offset_in_result_2);
  auto expected =
    cudf::concatenate(std::vector<cudf::table_view>{expected1->view(), expected2->view()});
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, concatenateSingleRowGroupIfCompressed)
{
  SetUp();
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 148, 157);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));
  // Execute the read task
  task.execute();
  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result     = 15;
  constexpr cudf::size_type start_offset_in_result = 145;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, readWithSmDecompressBackendEnabled)
{
  SetUp();
  // Configure compression and explicitly request SM decompression backend.
  _query_ctx->parameters.compression_format           = gqe::compression_format::lz4;
  _query_ctx->parameters.secondary_compression_format = gqe::compression_format::cascaded;
  _query_ctx->parameters.decompression_backend        = gqe::decompression_backend::sm;

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  task.execute();

  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), _input_tables[0]);
}

TEST_F(InMemoryReadTaskTest, readWithDeDecompressBackendEnabled)
{
  SetUp();
  // Configure compression and explicitly request DE decompression backend.
  _query_ctx->parameters.compression_format           = gqe::compression_format::lz4;
  _query_ctx->parameters.secondary_compression_format = gqe::compression_format::cascaded;
  _query_ctx->parameters.decompression_backend        = gqe::decompression_backend::de;

  auto const device_supports_de =
    gqe::device_properties::instance().get<gqe::device_properties::property::memDecompressSupport>(
      rmm::get_current_cuda_device());

  if (!device_supports_de) {
    ASSERT_THROW(create_row_groups(_input_tables.begin(), _input_tables.begin() + 1),
                 std::invalid_argument);
    return;
  }

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  task.execute();

  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), _input_tables[0]);
}

TEST_F(InMemoryReadTaskTest, PruningWithoutCompression)
{
  SetUp(1024 * 1024, 64 * 1024);
  // Compress input dat
  // _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 148, 157);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_FALSE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result     = 64 * 1024;
  constexpr cudf::size_type start_offset_in_result = 0;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, PruningWithCompression)
{
  SetUp(2 * 64 * 1024, 64 * 1024);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 148, 157);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result     = 64 * 1024;
  constexpr cudf::size_type start_offset_in_result = 0;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, DecompressionNoKernelAPI)
{
  // we use the PruningWithCompression test to test if the kernel API is working
  SetUp(2 * 64 * 1024, 64 * 1024, false, NUM_TABLES, true);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 148, 157);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result     = 64 * 1024;
  constexpr cudf::size_type start_offset_in_result = 0;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, PruningWithCompressionAndTwoFilters)
{
  SetUp(4 * 64 * 1024, 64 * 1024);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter =
    create_two_range_filters(0, 100, 157, 3 * 64 * 1024, 3 * 64 * 1024 + 100);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result = 64 * 1024;
  auto partition1 = create_input_table(num_rows_in_result, 0 /*offset*/);
  auto partition2 = create_input_table(num_rows_in_result, 3 * 64 * 1024 /*offset*/);
  auto expected =
    cudf::concatenate(std::vector<cudf::table_view>{partition1->view(), partition2->view()});
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, PruningWithCompressionOverlap)
{
  SetUp(4 * 64 * 1024, 64 * 1024);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 100, 65 * 1024);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result = 128 * 1024;
  auto expected = create_input_table(num_rows_in_result, 0 /*offset*/);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, PruningWithCompressionOverlapAndStringColumn)
{
  SetUp(4 * 64 * 1024, 64 * 1024, true);
  // SetUp(1024, 512, true);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 100, 65 * 1024);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result = 128 * 1024;
  auto expected = create_input_table(num_rows_in_result, 0 /*offset*/, true /*add_string_column*/);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest,
       PruningWithCompressionOverlapAndStringColumnWithGapSingleRowGroupMultiplePartitions)
{
  SetUp(4 * 64 * 1024, 64 * 1024, true);
  // SetUp(1024, 512, true);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter =
    create_two_range_filters(0, 100, 157, 3 * 64 * 1024, 3 * 64 * 1024 + 100);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.

  constexpr cudf::size_type num_rows_in_result = 64 * 1024;
  auto col_view = cudf::strings_column_view{_input_tables[0].column(4)};
  int32_t partition2_char_offset;
  cudaMemcpy(&partition2_char_offset,
             col_view.offsets().data<int32_t>() + num_rows_in_result * 3,
             sizeof(int32_t),
             cudaMemcpyDefault);

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  auto partition1 =
    create_input_table(num_rows_in_result, 0 /*offset*/, true /*add_string_column*/);

  auto partition2 = create_input_table(num_rows_in_result,
                                       3 * 64 * 1024 /*offset*/,
                                       true /*add_string_column*/,
                                       partition2_char_offset);

  auto expected =
    cudf::concatenate(std::vector<cudf::table_view>{partition1->view(), partition2->view()});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *task.result());
}

TEST_F(InMemoryReadTaskTest, PruningWithCompressionOverlapAndStringColumnWithGapmultiplierowGroups)
{
  constexpr auto partition_size         = 64 * 1024;
  constexpr auto num_rows_per_row_group = 4 * 64 * 1024;
  SetUp(num_rows_per_row_group, partition_size, true);
  // SetUp(1024, 512, true);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter consisting selecting two partitions:
  // [148, 157) selects the first partition in the first row group
  // [start_offset2, start_offset2 + 100) selects the second partition in the second row group
  constexpr auto start_offset2 = num_rows_per_row_group + partition_size;
  std::unique_ptr<gqe::expression> partial_filter =
    create_two_range_filters(0, 100, 157, start_offset2, start_offset2 + 100);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.

  // Get the start character for the second result partition
  auto col_view = cudf::strings_column_view{_input_tables[1].column(4)};
  int32_t partition2_char_offset;
  cudaMemcpy(&partition2_char_offset,
             col_view.offsets().data<int32_t>() + partition_size * 1,
             sizeof(int32_t),
             cudaMemcpyDefault);

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  auto partition1 = create_input_table(partition_size, 0 /*offset*/, true /*add_string_column*/);

  auto partition2 = create_input_table(
    partition_size, start_offset2 /*offset*/, true /*add_string_column*/, partition2_char_offset);

  auto expected =
    cudf::concatenate(std::vector<cudf::table_view>{partition1->view(), partition2->view()});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *task.result());
}

TEST_F(InMemoryReadTaskTest, SlicedCompressionOverlapAndStringColumnWithNoFilter)
{
  SetUp(128 * 1024, 64 * 1024, false, 4);
  // SetUp(1024, 512, true);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  // 4 table, each is 128 * 1024 rows concatenated with no filter
  auto expected = create_input_table(4 * 128 * 1024, 0 /*offset*/, false /*add_string_column*/);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *task.result());
}

TEST_F(InMemoryReadTaskTest, PruningWithCompressedSlicedColumnSmallWithoutCompression)
{
  SetUp(5, 64 * 1024, true);
  // SetUp(1024, 512, true);
  // Compress input dat
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Create a zone map filter which removes all partitions
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 0, 3);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result

  size_t num_rows_in_result     = 5;
  size_t start_offset_in_result = 0;
  auto expected =
    create_input_table(num_rows_in_result, start_offset_in_result, true /*add_string_column*/);
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *task.result());
}

TEST_F(InMemoryReadTaskTest, PruningWithCompressionOverlapAndStringColumnAndCascaded)
{
  SetUp(4 * 64 * 1024, 64 * 1024, true);
  // Compress input data.
  _query_ctx->parameters.compression_format           = gqe::compression_format::ans;
  _query_ctx->parameters.secondary_compression_format = gqe::compression_format::cascaded;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 100, 65 * 1024);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.

  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result = 128 * 1024;
  auto expected = create_input_table(num_rows_in_result, 0 /*offset*/, true /*add_string_column*/);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

// Regression test for https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/merge_requests/471
TEST_F(InMemoryReadTaskTest, SecondaryFormatLz4WithCpu)
{
  SetUp(4 * 64 * 1024, 64 * 1024);
  _query_ctx->parameters.compression_format           = gqe::compression_format::cascaded;
  _query_ctx->parameters.secondary_compression_format = gqe::compression_format::lz4;
  _query_ctx->parameters.use_cpu_compression          = true;

  auto row_groups = create_row_groups(_input_tables.begin(), _input_tables.end());
  auto task       = create_read_task(row_groups, _column_indexes, _data_types);
  task.execute();

  ASSERT_TRUE(task.result().has_value());
  auto expected = cudf::concatenate(_input_tables);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, BitpackingAsSecondaryFormat)
{
  constexpr cudf::size_type num_rows                = 64 * 1024;
  constexpr int32_t compression_chunk_size_in_bytes = 4 * 1024;
  SetUp(num_rows, num_rows, false /*use_string_column*/, 0 /*num_tables*/);
  _query_ctx->parameters.compression_format           = gqe::compression_format::lz4;
  _query_ctx->parameters.secondary_compression_format = gqe::compression_format::bitpacking;
  _query_ctx->parameters.compression_chunk_size       = compression_chunk_size_in_bytes;
  _query_ctx->parameters.compression_ratio_threshold  = 1.0;
  _query_ctx->parameters.secondary_compression_ratio_threshold      = 1.5;
  _query_ctx->parameters.secondary_compression_multiplier_threshold = 1.5;
  constexpr int64_t bitpacking_base_value                           = 1024;
  constexpr cudf::size_type bitpacking_period                       = 1024;
  std::vector<int64_t> bitpacking_pattern(bitpacking_period);
  std::iota(bitpacking_pattern.begin(), bitpacking_pattern.end(), bitpacking_base_value);
  std::mt19937 random_engine{42};
  std::shuffle(bitpacking_pattern.begin(), bitpacking_pattern.end(), random_engine);

  std::vector<int64_t> bitpacking_values(num_rows);
  for (cudf::size_type row_idx = 0; row_idx < num_rows; ++row_idx) {
    bitpacking_values[row_idx] = bitpacking_pattern[row_idx % bitpacking_period];
  }
  std::vector<int64_t> lz4_values(num_rows, int64_t{0});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>(bitpacking_values.begin(),
                                                                       bitpacking_values.end())
                         .release());
  columns.emplace_back(
    cudf::test::fixed_width_column_wrapper<int64_t>(lz4_values.begin(), lz4_values.end())
      .release());
  auto expected =
    std::make_unique<cudf::table>(std::move(columns), cudf::get_default_stream(), _memory_resource);
  auto const expected_view = expected->view();

  _input_tables.push_back(expected_view);
  _tables.push_back(std::move(expected));
  _column_indexes = {0, 1};
  _data_types     = {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT64)};

  auto row_groups = create_row_groups(_input_tables.begin(), _input_tables.end());
  auto const* bitpacking_column =
    dynamic_cast<gqe::storage::compressed_sliced_column_interface const*>(
      &row_groups.front()->get_column(0));
  auto const* lz4_column = dynamic_cast<gqe::storage::compressed_sliced_column_interface const*>(
    &row_groups.front()->get_column(1));
  ASSERT_NE(bitpacking_column, nullptr);
  ASSERT_NE(lz4_column, nullptr);
  EXPECT_EQ(bitpacking_column->data_buffer_view().compression_format(),
            gqe::compression_format::bitpacking);
  EXPECT_EQ(lz4_column->data_buffer_view().compression_format(), gqe::compression_format::lz4);

  auto task = create_read_task(row_groups, _column_indexes, _data_types);
  task.execute();

  ASSERT_TRUE(task.result().has_value());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected_view);
}

TEST_F(InMemoryReadTaskTest, IneffectiveCompressionStats)
{
  auto const num_rows = 2 * 64 * 1024;
  SetUp(num_rows, 64 * 1024);
  // Compress input data.
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Column with compression ratio ~1.41 would not take effect, fall back to uncompressed.
  _query_ctx->parameters.compression_ratio_threshold = 2.0;
  // Create a zone map filter 148 <= col0 < 157, which returns 1 partition.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 148, 157);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  // All of the row groups should be uncompressed due to ineffective compression ratio.
  for (const auto& row_group : row_groups) {
    for (int64_t i = 0; i < row_group->num_columns(); ++i) {
      ASSERT_FALSE(row_group->get_column(i).is_compressed());
      ASSERT_TRUE(row_group->get_column(i).get_uncompressed_size() == num_rows * sizeof(int32_t));
      ASSERT_TRUE(row_group->get_column(i).get_compressed_size() ==
                  row_group->get_column(i).get_uncompressed_size());
      ASSERT_TRUE(row_group->get_column(i).get_compression_ratio() == 1.0);
      auto stats = std::get<gqe::fixed_width_compression_statistics>(
        row_group->get_column(i).get_compression_stats());
      ASSERT_TRUE(stats.num_compressed_row_groups == 0);
      ASSERT_TRUE(stats.compressed_size == stats.uncompressed_size);
      // The primary compressed size is smaller even if the compression isineffective.
      ASSERT_TRUE(stats.primary_compressed_size < stats.compressed_size);
      ASSERT_TRUE(stats.secondary_compressed_size == 0);
      ASSERT_TRUE(stats.num_primary_compressed_row_groups == 0);
      ASSERT_TRUE(stats.num_secondary_compressed_row_groups == 0);
    }
  }
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result     = 64 * 1024;
  constexpr cudf::size_type start_offset_in_result = 0;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, BitpackingFallsBackToUncompressed)
{
  SetUp(2 * 64 * 1024, 64 * 1024);
  _query_ctx->parameters.compression_format          = gqe::compression_format::bitpacking;
  _query_ctx->parameters.compression_ratio_threshold = 1000.0;

  auto row_groups = create_row_groups(_input_tables.begin(), _input_tables.end());
  for (auto const* row_group : row_groups) {
    for (int64_t i = 0; i < row_group->num_columns(); ++i) {
      EXPECT_FALSE(row_group->get_column(i).is_compressed());
    }
  }

  auto task = create_read_task(row_groups, _column_indexes, _data_types);
  task.execute();

  ASSERT_TRUE(task.result().has_value());
  auto expected = cudf::concatenate(_input_tables);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, EffectiveCompressionStats)
{
  auto const num_rows = 2 * 64 * 1024;
  SetUp(num_rows, 64 * 1024);
  // Compress input data.
  _query_ctx->parameters.compression_format = gqe::compression_format::ans;
  // Column with compression ratio ~1.41 would take effect.
  _query_ctx->parameters.compression_ratio_threshold = 1.0;
  // Create a zone map filter 148 <= col0 < 157, which returns 1 partition.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 148, 157);
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  // All of the row groups should be compressed due to effective compression ratio.
  for (const auto& row_group : row_groups) {
    for (int64_t i = 0; i < row_group->num_columns(); ++i) {
      ASSERT_TRUE(row_group->get_column(i).is_compressed());
      ASSERT_TRUE(row_group->get_column(i).get_uncompressed_size() == num_rows * sizeof(int32_t));
      ASSERT_TRUE(row_group->get_column(i).get_compressed_size() <
                  row_group->get_column(i).get_uncompressed_size());
      ASSERT_TRUE(row_group->get_column(i).get_compression_ratio() > 1.0);
      auto stats = std::get<gqe::fixed_width_compression_statistics>(
        row_group->get_column(i).get_compression_stats());
      ASSERT_TRUE(stats.num_compressed_row_groups == 1);
      ASSERT_TRUE(stats.compressed_size < stats.uncompressed_size);
      ASSERT_TRUE(stats.primary_compressed_size == stats.compressed_size);
      ASSERT_TRUE(stats.secondary_compressed_size == 0);
      ASSERT_TRUE(stats.num_primary_compressed_row_groups == 1);
      ASSERT_TRUE(stats.num_secondary_compressed_row_groups == 0);
    }
  }

  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());

  constexpr cudf::size_type num_rows_in_result     = 64 * 1024;
  constexpr cudf::size_type start_offset_in_result = 0;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, DuplicateSecondaryCompressionFormatThrows)
{
  auto const num_rows = 2 * 64 * 1024;
  SetUp(num_rows, 64 * 1024);
  _query_ctx->parameters.compression_format           = gqe::compression_format::ans;
  _query_ctx->parameters.secondary_compression_format = gqe::compression_format::ans;
  _query_ctx->parameters.compression_ratio_threshold  = 1.0;

  EXPECT_THROW(create_row_groups(_input_tables.begin(), _input_tables.end()),
               std::invalid_argument);
}

TEST_F(InMemoryReadTaskTest, keepResultIfDummyCopyEnabled)
{
  SetUp();
  // Verify that enabling in-memory dummy copy only changes copy workload (extra memcpy passes),
  // but does not change result semantics: the task should still return an owned result and
  // preserve the expected table contents.

  // Disable zero-copying
  _query_ctx->parameters.read_use_zero_copy = false;
  // Enable dummy copy
  _query_ctx->parameters.in_memory_dummy_copy_multiplier = 2.5;

  // Create a task with a single row group
  std::vector<const gqe::storage::row_group_interface*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  gqe::storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  ASSERT_TRUE(task.result().has_value());
  EXPECT_TRUE(*task.is_result_owned());
  cudf::table_view expected = _input_tables[0];
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected);
}
