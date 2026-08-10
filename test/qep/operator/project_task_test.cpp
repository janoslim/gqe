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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

using cudf::test::fixed_width_column_wrapper;

class private_state final : public gqe::qep::task_private_state {};

class QepProjectTaskTest : public gqe::test::BaseFixture {
 protected:
  QepProjectTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::qep::state_container run(
    gqe::qep::project_task const& task,
    gqe::qep::state_container_view input,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    auto result = task.next(input, ctx_ref, stream, mr);
    if (!result.has_value()) {
      ADD_FAILURE() << "project task returned no result";
      return {};
    }
    return std::move(*result);
  }

  gqe::context_reference ctx_ref;
};

TEST_F(QepProjectTaskTest, ReordersAndDuplicatesExactOwnedAndBorrowedStates)
{
  auto owned = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{fixed_width_column_wrapper<int64_t>{1, 2, 3}.release()});
  fixed_width_column_wrapper<int64_t> borrowed_owner{10, 20, 30};
  auto borrowed =
    gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{borrowed_owner});
  gqe::qep::state_container input{owned, borrowed};
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));

  auto result = run(gqe::qep::project_task{std::move(expressions)}, input);

  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], borrowed);
  EXPECT_EQ(result[1], owned);
  EXPECT_EQ(result[2], borrowed);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column_view>(*result[0]));
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*result[1]));
}

TEST_F(QepProjectTaskTest, MixedProjectionComputesOwnedResultsInOutputOrder)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{1, 2, 3}.release()})
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{10, 20, 30}.release()})
                 .build();
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(5)));
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                          std::make_shared<gqe::literal_expression<int64_t>>(7)));
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  auto result = run(gqe::qep::project_task{std::move(expressions)}, input);

  ASSERT_EQ(result.size(), 4u);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*result[0]));
  EXPECT_EQ(result[1], input[1]);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*result[2]));
  EXPECT_EQ(result[3], input[0]);
  auto table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  fixed_width_column_wrapper<int64_t> expected_first_computed{6, 7, 8};
  fixed_width_column_wrapper<int64_t> expected_first_reference{10, 20, 30};
  fixed_width_column_wrapper<int64_t> expected_second_computed{17, 27, 37};
  fixed_width_column_wrapper<int64_t> expected_second_reference{1, 2, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_first_computed);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_first_reference);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(2), expected_second_computed);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(3), expected_second_reference);
}

TEST_F(QepProjectTaskTest, ComputedExpressionPropagatesInputNulls)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{{1, 2, 3}, {true, false, true}}.release()})
                 .build();
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(5)));
  auto result = run(gqe::qep::project_task{std::move(expressions)}, input);

  auto table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  fixed_width_column_wrapper<int64_t> expected{{6, 0, 8}, {true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(table->column(0), expected);
}

TEST_F(QepProjectTaskTest, EmptyRowsReturnSchemaBearingProjection)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::cudf_column{
                   cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64})})
                 .build();
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(1)));
  auto result = run(gqe::qep::project_task{std::move(expressions)}, input);

  auto table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->num_columns(), 2);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(result[0], input[0]);
}

TEST_F(QepProjectTaskTest, SlicedReferenceKeepsIdentityAndComputedAlignment)
{
  fixed_width_column_wrapper<int64_t> owner{0, 1, 2, 3, 4};
  auto const sliced = cudf::slice(owner, {1, 4}).front();
  auto slot         = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  gqe::qep::state_container input{slot};
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(10)));
  auto result = run(gqe::qep::project_task{std::move(expressions)}, input);

  EXPECT_EQ(result[0], slot);
  auto table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->column(0).offset(), sliced.offset());
  fixed_width_column_wrapper<int64_t> expected{11, 12, 13};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected);
}

TEST_F(QepProjectTaskTest, MaskedReferenceOnlyPassesMaskAndColumnsWithoutAllocation)
{
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{
        fixed_width_column_wrapper<bool>{true, false, true}.release()})
      .add_state(
        gqe::qep::state_kind::cudf_column{fixed_width_column_wrapper<int64_t>{1, 2, 3}.release()})
      .add_state(
        gqe::qep::state_kind::cudf_column{fixed_width_column_wrapper<int64_t>{4, 5, 6}.release()})
      .build();
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));

  rmm::mr::statistics_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  auto result = run(
    gqe::qep::project_task{std::move(expressions)}, input, cudf::get_default_stream(), result_mr);

  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], input[0]);
  EXPECT_EQ(result[1], input[2]);
  EXPECT_EQ(result[2], input[1]);
  EXPECT_EQ(result_mr.get_allocations_counter().total, 0);
}

TEST_F(QepProjectTaskTest, MaskedProjectionPreservesOriginalAndInjectedNullSemantics)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::valid_mask{
                   fixed_width_column_wrapper<bool>{true, false, true}.release()})
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{{1, 2, 3}, {false, true, true}}.release()})
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{10, 20, 30}.release()})
                 .build();
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(5)));
  auto result = run(gqe::qep::project_task{std::move(expressions)}, input);

  auto masked = gqe::qep::masked_table_view::try_from(result);
  ASSERT_TRUE(masked.has_value());
  EXPECT_EQ(result[0], input[0]);
  EXPECT_EQ(result[1], input[2]);
  fixed_width_column_wrapper<int64_t> expected{{0, 0, 8}, {false, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked->columns.column(1), expected);
}

TEST_F(QepProjectTaskTest, MaskedProjectionUsesExplicitStreamAndMemoryResource)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::valid_mask{
                   fixed_width_column_wrapper<bool>{true, false, true}.release()})
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{1, 2, 3}.release()})
                 .build();
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(10)));
  rmm::cuda_stream stream;
  rmm::mr::tracking_resource_adaptor mr{cudf::get_current_device_resource_ref()};
  auto result = run(gqe::qep::project_task{std::move(expressions)}, input, stream.view(), mr);
  stream.synchronize();

  auto masked = gqe::qep::masked_table_view::try_from(result);
  ASSERT_TRUE(masked.has_value());
  fixed_width_column_wrapper<int64_t> expected{{11, 0, 13}, {true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked->columns.column(0), expected);
  EXPECT_TRUE(
    mr.get_outstanding_allocations().contains(const_cast<void*>(masked->columns.column(0).head())));
}

TEST_F(QepProjectTaskTest, RejectsInvalidConstructionContextShapesAndReferences)
{
  EXPECT_THROW(gqe::qep::project_task({}), std::invalid_argument);
  std::vector<std::unique_ptr<gqe::expression>> null_expression;
  null_expression.push_back(nullptr);
  EXPECT_THROW(gqe::qep::project_task(std::move(null_expression)), std::invalid_argument);

  std::vector<std::unique_ptr<gqe::expression>> valid_expression;
  valid_expression.push_back(std::make_unique<gqe::column_reference_expression>(0));
  gqe::qep::project_task task{std::move(valid_expression)};
  gqe::qep::state_container empty;
  EXPECT_THROW((void)task.next(empty, ctx_ref), std::logic_error);
  auto count_only = gqe::qep::make_row_count_container(3);
  EXPECT_THROW((void)task.next(count_only, ctx_ref), std::logic_error);
  gqe::qep::state_container null_slot{nullptr};
  EXPECT_THROW((void)task.next(null_slot, ctx_ref), std::logic_error);
  gqe::qep::state_container private_slot{gqe::qep::make_shared_state(
    gqe::qep::state_kind::task_private{std::make_unique<private_state>()})};
  EXPECT_THROW((void)task.next(private_slot, ctx_ref), std::logic_error);

  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{1, 2}.release()})
                 .build();
  gqe::context_reference null_context{ctx_ref._task_manager_context, nullptr};
  EXPECT_THROW((void)task.next(input, null_context), std::logic_error);

  std::vector<std::unique_ptr<gqe::expression>> bad_reference;
  bad_reference.push_back(std::make_unique<gqe::column_reference_expression>(1));
  gqe::qep::project_task out_of_range{std::move(bad_reference)};
  EXPECT_THROW((void)out_of_range.next(input, ctx_ref), std::logic_error);
}

TEST_F(QepProjectTaskTest, CloneIsIndependentAndEquivalent)
{
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(5)));
  auto original   = std::make_unique<gqe::qep::project_task>(std::move(expressions));
  auto clone_task = original->clone();
  original.reset();
  auto* clone = dynamic_cast<gqe::qep::project_task*>(clone_task.get());
  ASSERT_NE(clone, nullptr);
  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::cudf_column{
                   fixed_width_column_wrapper<int64_t>{1, 2, 3}.release()})
                 .build();

  auto result = run(*clone, input);
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  fixed_width_column_wrapper<int64_t> expected{6, 7, 8};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected);
}
