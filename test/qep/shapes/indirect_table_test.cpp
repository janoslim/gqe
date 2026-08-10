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

#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

namespace {

gqe::qep::shared_state owned_int64(std::initializer_list<int64_t> values)
{
  return gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper(values).release()});
}

gqe::qep::position_list_handle positions(std::initializer_list<int32_t> values)
{
  return gqe::qep::make_position_list(int32_column_wrapper(values).release());
}

gqe::qep::indirect_gather_source one_column_gather(
  gqe::qep::shared_state source,
  gqe::qep::position_list_handle map,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  auto relation = gqe::qep::indirect_table_source_state::make({std::move(source)});
  return gqe::qep::indirect_gather_source_state::make(std::move(relation), std::move(map), policy);
}

gqe::qep::shared_state indirect(gqe::qep::indirect_gather_source gather,
                                cudf::size_type source_column_index = 0)
{
  return gqe::qep::indirect_column_state::make(std::move(gather), source_column_index);
}

}  // namespace

TEST(QepPositionListState, ValidatesStorageTypeAndNullability)
{
  auto valid = positions({2, 0, 2});
  EXPECT_EQ(valid->column().size(), 3);

  auto wrapped        = gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{valid});
  auto const& generic = std::get<gqe::qep::state_kind::position_list>(*wrapped);
  EXPECT_EQ(generic.state(), valid);

  auto wrong_type = int64_column_wrapper{0, 1}.release();
  EXPECT_THROW(std::ignore = gqe::qep::make_position_list(std::move(wrong_type)),
               std::invalid_argument);
  auto nullable = int32_column_wrapper{{0, 1}, {true, false}}.release();
  EXPECT_THROW(std::ignore = gqe::qep::make_position_list(std::move(nullable)),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::make_position_list(nullptr), std::invalid_argument);
  EXPECT_THROW(
    std::ignore = gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{nullptr}),
    std::invalid_argument);
}

TEST(QepStateColumnView, ConvertsOwnedBorrowedAndSlicedColumns)
{
  auto owned      = owned_int64({1, 2, 3});
  auto owned_view = gqe::qep::to_column_view(owned);
  ASSERT_TRUE(owned_view.has_value());
  EXPECT_EQ(owned_view->head(),
            std::get<gqe::qep::state_kind::cudf_column>(*owned).column->view().head());

  int64_column_wrapper owner{0, 10, 20, 30};
  auto sliced        = cudf::slice(owner, {1, 4}).front();
  auto borrowed      = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  auto borrowed_view = gqe::qep::to_column_view(borrowed);
  ASSERT_TRUE(borrowed_view.has_value());
  EXPECT_EQ(borrowed_view->head(), sliced.head());
  EXPECT_EQ(borrowed_view->offset(), sliced.offset());
}

TEST(QepStateColumnView, RejectsWrongKindAndThrowsForMalformedStorage)
{
  auto wrong_kind = gqe::qep::make_shared_state(gqe::qep::state_kind::row_count{1});
  auto null_owned = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{nullptr});

  EXPECT_FALSE(gqe::qep::to_column_view(wrong_kind).has_value());
  EXPECT_THROW(std::ignore = gqe::qep::to_column_view(nullptr), std::logic_error);
  EXPECT_THROW(std::ignore = gqe::qep::to_column_view(null_owned), std::logic_error);
}

TEST(QepStateTableView, ThrowsForMalformedStorageAndRejectsWrongKinds)
{
  gqe::qep::state_container null_slot{nullptr};
  gqe::qep::state_container null_owned{
    gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{nullptr})};
  gqe::qep::state_container wrong_kind{gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()})};

  EXPECT_THROW(std::ignore = gqe::qep::to_table_view(null_slot), std::logic_error);
  EXPECT_THROW(std::ignore = gqe::qep::to_table_view(null_owned), std::logic_error);
  EXPECT_FALSE(gqe::qep::to_table_view(wrong_kind).has_value());
}

TEST(QepIndirectTableSource, AcceptsOwnedBorrowedSlicedNullableAndZeroRowColumns)
{
  auto owned = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{
    int64_column_wrapper{{1, 2, 3}, {true, false, true}}.release()});
  int32_column_wrapper owner{0, 10, 20, 30};
  auto sliced   = cudf::slice(owner, {1, 4}).front();
  auto borrowed = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});

  auto relation = gqe::qep::indirect_table_source_state::make({owned, borrowed});
  EXPECT_EQ(relation->row_count(), 3);
  EXPECT_EQ(relation->num_columns(), 2);
  auto view = relation->contiguous_table_view();
  EXPECT_EQ(view.column(0).null_count(), 1);
  EXPECT_EQ(view.column(1).head(), sliced.head());
  EXPECT_EQ(view.column(1).offset(), sliced.offset());

  auto empty_owned          = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{
    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64})});
  auto empty_borrowed_owner = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto empty_borrowed       = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column_view{empty_borrowed_owner->view()});
  auto empty_relation = gqe::qep::indirect_table_source_state::make({empty_owned, empty_borrowed});
  EXPECT_EQ(empty_relation->row_count(), 0);
  EXPECT_EQ(empty_relation->num_columns(), 2);
}

TEST(QepIndirectTableSource, RejectsEmptyNullNonConcreteAndMismatchedColumns)
{
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make({}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make({nullptr}),
               std::logic_error);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make(
                 {gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{nullptr})}),
               std::logic_error);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make(
                 {gqe::qep::make_shared_state(gqe::qep::state_kind::row_count{1})}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make(
                 {owned_int64({1, 2}), owned_int64({3})}),
               std::invalid_argument);
}

TEST(QepIndirectGatherSource, RejectsNullHandles)
{
  auto relation = gqe::qep::indirect_table_source_state::make({owned_int64({1})});
  EXPECT_THROW(std::ignore = gqe::qep::indirect_gather_source_state::make(nullptr, positions({0})),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_gather_source_state::make(relation, nullptr),
               std::invalid_argument);
}

TEST(QepIndirectColumnState, ValidatesIndexAndSharesRelationGatherAndPositionLifetimes)
{
  auto first_source  = owned_int64({10, 20, 30});
  auto second_source = owned_int64({100, 200, 300});
  auto map           = positions({2, 0});
  std::weak_ptr<gqe::qep::state_kind::type> first_source_weak{first_source};
  std::weak_ptr<gqe::qep::state_kind::type> second_source_weak{second_source};
  std::weak_ptr<gqe::qep::position_list_state const> map_weak{map};

  auto relation = gqe::qep::indirect_table_source_state::make({first_source, second_source});
  auto gather   = gqe::qep::indirect_gather_source_state::make(relation, map);
  std::weak_ptr<gqe::qep::indirect_table_source_state const> relation_weak{relation};
  std::weak_ptr<gqe::qep::indirect_gather_source_state const> gather_weak{gather};

  EXPECT_THROW(std::ignore = gqe::qep::indirect_column_state::make(nullptr, 0),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_column_state::make(gather, -1),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_column_state::make(gather, 2),
               std::invalid_argument);

  auto first  = indirect(gather, 0);
  auto second = indirect(gather, 1);
  first_source.reset();
  second_source.reset();
  map.reset();
  relation.reset();
  gather.reset();

  EXPECT_FALSE(first_source_weak.expired());
  EXPECT_FALSE(second_source_weak.expired());
  EXPECT_FALSE(map_weak.expired());
  EXPECT_FALSE(relation_weak.expired());
  EXPECT_FALSE(gather_weak.expired());
  {
    auto const& first_state  = *std::get<gqe::qep::state_kind::indirect_column>(*first).state();
    auto const& second_state = *std::get<gqe::qep::state_kind::indirect_column>(*second).state();
    EXPECT_EQ(first_state.gather_source(), second_state.gather_source());
    EXPECT_EQ(first_state.gather_source()->position_list(),
              second_state.gather_source()->position_list());
    EXPECT_EQ(first_state.source_column_index(), 0);
    EXPECT_EQ(second_state.source_column_index(), 1);
    EXPECT_EQ(first_state.gather_source()->relation_source()->num_columns(), 2);
    EXPECT_EQ(first_state.gather_source()->position_list().get(), map_weak.lock().get());
  }
  first.reset();
  second.reset();
  EXPECT_TRUE(first_source_weak.expired());
  EXPECT_TRUE(second_source_weak.expired());
  EXPECT_TRUE(map_weak.expired());
  EXPECT_TRUE(relation_weak.expired());
  EXPECT_TRUE(gather_weak.expired());
}

TEST(QepIndirectColumnState, PublicWrapperOnlyExposesAnImmutableHandle)
{
  using wrapper = gqe::qep::state_kind::indirect_column;
  using handle  = std::remove_cvref_t<decltype(std::declval<wrapper const&>().state())>;
  static_assert(!std::is_aggregate_v<wrapper>);
  static_assert(std::is_const_v<typename handle::element_type>);
  EXPECT_THROW(std::ignore = wrapper{nullptr}, std::invalid_argument);
}

TEST(QepIndirectColumnState, BorrowedSlicedSourcePreservesViewWhileExternalOwnerLives)
{
  int64_column_wrapper owner{0, 10, 20, 30};
  auto sliced = cudf::slice(owner, {1, 4}).front();
  auto source = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  auto gather = one_column_gather(source, positions({2, 0}), cudf::out_of_bounds_policy::NULLIFY);
  source.reset();

  // `owner` must remain alive while the borrowed view is used. Destroying it before inspecting the
  // view would make the access undefined, so external ownership remains an API contract.
  auto view = gather->relation_source()->contiguous_table_view().column(0);
  EXPECT_EQ(view.head(), sliced.head());
  EXPECT_EQ(view.offset(), sliced.offset());
  EXPECT_EQ(gather->bounds_policy(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST(QepIndirectTableView, RecognizesIndirectMixedAndMaskedShapesByIdentity)
{
  auto gather    = one_column_gather(owned_int64({10, 20, 30}), positions({1, 0}));
  auto projected = indirect(gather);

  gqe::qep::state_container indirect_only{projected};
  auto one = gqe::qep::indirect_table_view::try_from(indirect_only);
  ASSERT_TRUE(one.has_value());
  EXPECT_EQ(one->row_count, 2);
  EXPECT_FALSE(one->mask.has_value());
  ASSERT_EQ(one->indirect_columns.size(), 1);
  EXPECT_EQ(one->indirect_columns.front().output_index, 0);
  EXPECT_EQ(one->indirect_columns.front().gather_source, gather);
  EXPECT_EQ(one->indirect_columns.front().source_column_index, 0);

  gqe::qep::state_container mixed{owned_int64({1, 2}), projected, owned_int64({3, 4})};
  auto three = gqe::qep::indirect_table_view::try_from(mixed);
  ASSERT_TRUE(three.has_value());
  EXPECT_EQ(three->columns.size(), 3);
  EXPECT_EQ(three->indirect_columns.front().output_index, 1);

  gqe::qep::state_container masked{gqe::qep::make_shared_state(gqe::qep::state_kind::valid_mask{
                                     bool_column_wrapper{true, false}.release()}),
                                   projected,
                                   owned_int64({5, 6})};
  auto masked_view = gqe::qep::indirect_table_view::try_from(masked);
  ASSERT_TRUE(masked_view.has_value());
  ASSERT_TRUE(masked_view->mask.has_value());
  EXPECT_EQ(masked_view->mask->type().id(), cudf::type_id::BOOL8);
  ASSERT_EQ(masked_view->indirect_columns.size(), 1);
  EXPECT_EQ(masked_view->indirect_columns.front().output_index, 0);
}

TEST(QepIndirectTableView, NonIndirectShapesReturnNullopt)
{
  gqe::qep::state_container empty;
  EXPECT_FALSE(gqe::qep::indirect_table_view::try_from(empty).has_value());

  gqe::qep::state_container plain{owned_int64({1})};
  EXPECT_FALSE(gqe::qep::indirect_table_view::try_from(plain).has_value());

  gqe::qep::state_container raw_positions{
    gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{positions({0})})};
  EXPECT_FALSE(gqe::qep::indirect_table_view::try_from(raw_positions).has_value());
}

TEST(QepIndirectTableView, RejectsMalformedClaimedShapes)
{
  auto projected = indirect(one_column_gather(owned_int64({10, 20}), positions({1, 0})));

  gqe::qep::state_container null_slot{projected, nullptr};
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_view::try_from(null_slot), std::logic_error);

  gqe::qep::state_container mismatched_direct{projected, owned_int64({1})};
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_view::try_from(mismatched_direct),
               std::logic_error);

  gqe::qep::state_container mismatched_mask{
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()}),
    projected};
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_view::try_from(mismatched_mask),
               std::logic_error);
}

TEST(QepIndirectTableView, LogicalGeometryUsesPositionListInsteadOfSourceRows)
{
  auto projected = indirect(one_column_gather(owned_int64({10, 20, 30}), positions({2, 0})));
  gqe::qep::state_container table{projected};
  auto view = gqe::qep::indirect_table_view::try_from(table);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 2);
  EXPECT_EQ(view->indirect_columns.front().gather_source->relation_source()->row_count(), 3);
  EXPECT_EQ(gqe::qep::get_row_count(table), 2);
}

TEST(QepIndirectState, GenericHelpersHandleIndirectAndPositionListStates)
{
  auto projected = indirect(one_column_gather(owned_int64({10, 20, 30}), positions({2, 0})));
  gqe::qep::state_container table{projected};
  EXPECT_FALSE(gqe::qep::to_table_view(table).has_value());

  gqe::qep::state_container raw_positions{
    gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{positions({2, 0, 2})})};
  EXPECT_EQ(gqe::qep::get_row_count(raw_positions), 3);
  EXPECT_FALSE(gqe::qep::to_table_view(raw_positions).has_value());
}
