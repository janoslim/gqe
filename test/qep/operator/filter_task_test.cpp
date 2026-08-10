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
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

std::unique_ptr<gqe::expression> greater_than(int64_t threshold)
{
  return std::make_unique<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(0),
    std::make_shared<gqe::literal_expression<int64_t>>(threshold));
}

cudf::table one_column_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return cudf::table{std::move(columns)};
}

class QepFilterTaskTest : public gqe::test::BaseFixture {
 protected:
  QepFilterTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::qep::masked_table_view run(
    gqe::qep::filter_task const& task,
    gqe::qep::state_container_view input,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    result = task.next(input, ctx_ref, stream, mr);
    EXPECT_TRUE(result.has_value());
    auto masked = gqe::qep::masked_table_view::try_from(*result);
    EXPECT_TRUE(masked.has_value());
    return *masked;
  }

  gqe::context_reference ctx_ref;
  std::optional<gqe::qep::state_container> result;
};

TEST_F(QepFilterTaskTest, ComputedPredicateProducesMaskAndPassesColumnsThrough)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper{1, 2, 3, 4}.release());
  columns.push_back(int64_column_wrapper{10, 20, 30, 40}.release());
  auto input =
    gqe::qep::state_container_builder().add_state(cudf::table{std::move(columns)}).build();
  auto const* first_slot  = input[0].get();
  auto const* second_slot = input[1].get();

  gqe::qep::filter_task task{greater_than(2)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, false, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
  EXPECT_EQ((*result)[1].get(), first_slot);
  EXPECT_EQ((*result)[2].get(), second_slot);
  EXPECT_EQ(masked.columns.num_columns(), 2);
}

TEST_F(QepFilterTaskTest, BareBooleanReferenceProducesOwnedIndependentMask)
{
  auto bools            = bool_column_wrapper{true, false, true}.release();
  auto const input_head = bools->view().head();
  auto input            = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::cudf_column{std::move(bools)})
                 .build();

  gqe::qep::filter_task task{std::make_unique<gqe::column_reference_expression>(0)};
  auto masked = run(task, input);

  bool_column_wrapper expected{true, false, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
  EXPECT_NE(masked.mask.head(), input_head);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::valid_mask>(*(*result)[0]));
}

TEST_F(QepFilterTaskTest, PreservesMixedOwnedAndBorrowedStateKinds)
{
  auto owned = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()});
  int64_column_wrapper borrowed_owner{4, 5, 6};
  auto borrowed =
    gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{borrowed_owner});
  gqe::qep::state_container input{owned, borrowed};

  gqe::qep::filter_task task{greater_than(0)};
  run(task, input);

  EXPECT_EQ((*result)[1], owned);
  EXPECT_EQ((*result)[2], borrowed);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*(*result)[1]));
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column_view>(*(*result)[2]));
}

TEST_F(QepFilterTaskTest, EmptyRowsRemainSchemaBearing)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
  auto input =
    gqe::qep::state_container_builder().add_state(cudf::table{std::move(columns)}).build();

  gqe::qep::filter_task task{greater_than(0)};
  auto masked = run(task, input);

  EXPECT_EQ(masked.row_count, 0);
  EXPECT_EQ(masked.columns.num_columns(), 1);
  EXPECT_EQ(result->size(), 2);
}

TEST_F(QepFilterTaskTest, AllFalsePredicatePreservesRowGeometry)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2, 3}.release()))
                 .build();
  gqe::qep::filter_task task{greater_than(100)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, false, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
  EXPECT_EQ(masked.row_count, 3);
}

TEST_F(QepFilterTaskTest, NullPredicateRowsDoNotSurviveMaterialization)
{
  auto values = int64_column_wrapper{{1, 2, 3, 4}, {true, false, true, false}}.release();
  auto input =
    gqe::qep::state_container_builder().add_state(one_column_table(std::move(values))).build();
  gqe::qep::filter_task task{greater_than(1)};
  auto masked = run(task, input);

  auto materialized = cudf::apply_boolean_mask(masked.columns, masked.mask);
  int64_column_wrapper expected{3};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(materialized->view().column(0), expected);
}

TEST_F(QepFilterTaskTest, SlicedBorrowedColumnKeepsAlignmentAndIdentity)
{
  int64_column_wrapper owner{0, 1, 2, 3, 4};
  auto const sliced = cudf::slice(owner, {1, 4}).front();
  auto slot         = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  gqe::qep::state_container input{slot};

  gqe::qep::filter_task task{greater_than(1)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked.mask, expected);
  EXPECT_EQ((*result)[1], slot);
  EXPECT_EQ(masked.columns.column(0).offset(), sliced.offset());
}

TEST_F(QepFilterTaskTest, ChainedMaskUsesNativeThreeValuedAnd)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::valid_mask{
                   bool_column_wrapper{{false, true, true}, {true, true, false}}.release()})
                 .add_state(one_column_table(int64_column_wrapper{-1, 2, 3}.release()))
                 .build();
  gqe::qep::filter_task task{greater_than(0)};
  auto masked = run(task, input);

  bool_column_wrapper expected{{false, true, false}, {true, true, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
  EXPECT_EQ((*result)[1], input[1]);
}

TEST_F(QepFilterTaskTest, MaskedOutNonNullRowCannotBeResurrectedByIsNull)
{
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{false, true, true}.release()})
      .add_state(one_column_table(int64_column_wrapper{{-1, 2, 3}, {true, false, true}}.release()))
      .build();
  auto condition = std::make_unique<gqe::is_null_expression>(
    std::make_shared<gqe::column_reference_expression>(0));
  gqe::qep::filter_task task{std::move(condition)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, true, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked.mask, expected);
}

TEST_F(QepFilterTaskTest, MaskedSliceRunsOnExplicitStreamAndResultResource)
{
  int64_column_wrapper owner{0, 1, 2, 3, 4};
  auto sliced = cudf::slice(owner, {1, 4}).front();
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()})
      .add_state(gqe::qep::state_kind::cudf_column_view{sliced})
      .build();
  gqe::qep::filter_task task{greater_than(1)};
  rmm::cuda_stream stream;
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto masked = run(task, input, stream.view(), result_mr);
  stream.synchronize();

  bool_column_wrapper expected{false, false, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked.mask, expected);
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(masked.mask.head())));
  result.reset();
}

TEST_F(QepFilterTaskTest, RejectsInvalidConstructionAndInputShapes)
{
  EXPECT_THROW(gqe::qep::filter_task{nullptr}, std::invalid_argument);
  gqe::qep::filter_task task{greater_than(0)};
  gqe::qep::state_container empty;
  EXPECT_THROW((void)task.next(empty, ctx_ref), std::logic_error);
  auto count_only = gqe::qep::make_row_count_container(3);
  EXPECT_THROW((void)task.next(count_only, ctx_ref), std::logic_error);

  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2}.release()))
                 .build();
  gqe::qep::filter_task non_boolean{std::make_unique<gqe::column_reference_expression>(0)};
  EXPECT_THROW((void)non_boolean.next(input, ctx_ref), std::logic_error);
}

TEST_F(QepFilterTaskTest, CloneIsIndependentAndEquivalent)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2, 3}.release()))
                 .build();
  auto original = std::make_unique<gqe::qep::filter_task>(greater_than(1));
  auto clone    = original->clone();
  original.reset();
  auto* cloned_filter = dynamic_cast<gqe::qep::filter_task*>(clone.get());
  ASSERT_NE(cloned_filter, nullptr);

  auto masked = run(*cloned_filter, input);
  bool_column_wrapper expected{false, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
}
