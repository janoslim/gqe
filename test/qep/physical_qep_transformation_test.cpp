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

#include <gqe/qep/physical_qep_transformation.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/perfect_hash_aggregate.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/operator/reduce_aggregate.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/chunked_parquet_writer.hpp>
#include <gqe/storage/oneshot_parquet_reader.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/table_provider.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace physical = gqe::physical;
namespace qep      = gqe::qep;
namespace storage  = gqe::storage;

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

cudf::data_type int64_type() { return cudf::data_type{cudf::type_id::INT64}; }

/**
 * @brief Minimal table provider backed by a name-to-table map.
 */
class map_table_provider : public storage::table_provider {
 public:
  void add_table(std::string name, std::shared_ptr<storage::table> table)
  {
    _tables.emplace(std::move(name), std::move(table));
  }

  [[nodiscard]] std::shared_ptr<storage::table> get_table(
    std::string_view table_name) const override
  {
    auto const it = _tables.find(std::string{table_name});
    if (it == _tables.end()) {
      throw std::logic_error("map_table_provider: unknown table " + std::string{table_name});
    }
    return it->second;
  }

 private:
  std::map<std::string, std::shared_ptr<storage::table>> _tables;
};

map_table_provider make_provider_with_parquet_table(std::string const& name)
{
  map_table_provider provider;
  provider.add_table(
    name, std::make_shared<storage::parquet_table>(std::vector<std::string>{"unused.parquet"}));
  return provider;
}

std::shared_ptr<physical::read_relation> make_read(
  std::string table_name,
  std::size_t num_columns,
  std::unique_ptr<gqe::expression> partial_filter,
  std::vector<std::shared_ptr<physical::relation>> subqueries = {})
{
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> data_types;
  for (std::size_t i = 0; i < num_columns; ++i) {
    column_names.push_back("col_" + std::to_string(i));
    data_types.push_back(int64_type());
  }
  return std::make_shared<physical::read_relation>(std::move(subqueries),
                                                   std::move(column_names),
                                                   std::move(table_name),
                                                   std::move(partial_filter),
                                                   std::move(data_types));
}

std::unique_ptr<gqe::expression> col_ref(cudf::size_type idx)
{
  return std::make_unique<gqe::column_reference_expression>(idx);
}

/**
 * @brief A simple `col(lhs_idx) < col(rhs_idx)` filter condition.
 */
std::unique_ptr<gqe::expression> less_condition(cudf::size_type lhs_idx, cudf::size_type rhs_idx)
{
  return std::make_unique<gqe::less_expression>(
    std::make_shared<gqe::column_reference_expression>(lhs_idx),
    std::make_shared<gqe::column_reference_expression>(rhs_idx));
}

std::vector<cudf::size_type> identity_indices(std::size_t count)
{
  std::vector<cudf::size_type> indices(count);
  for (std::size_t i = 0; i < count; ++i) {
    indices[i] = static_cast<cudf::size_type>(i);
  }
  return indices;
}

template <typename... Columns>
qep::state_container make_table(Columns&&... input_columns)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(sizeof...(input_columns));
  (columns.push_back(std::forward<Columns>(input_columns)), ...);
  return qep::state_container_builder().add_state(cudf::table{std::move(columns)}).build();
}

template <typename TaskType>
std::vector<qep::task const*> tasks_of_type(qep::query_execution_plan const& plan)
{
  std::vector<qep::task const*> matches;
  for (auto const* task : plan.tasks()) {
    if (dynamic_cast<TaskType const*>(task) != nullptr) { matches.push_back(task); }
  }
  return matches;
}

template <typename TaskType>
qep::task const* single_task_of_type(qep::query_execution_plan const& plan)
{
  auto const matches = tasks_of_type<TaskType>(plan);
  EXPECT_EQ(matches.size(), 1);
  return matches.empty() ? nullptr : matches.front();
}

}  // namespace

TEST(PhysicalQepTransformationTest, ReadLowersToSingleIterateTask)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);

  auto const plan = qep::physical_qep_transformation(read.get(), &provider);

  ASSERT_EQ(plan.tasks().size(), 1);
  auto const* read_task = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  ASSERT_NE(read_task, nullptr);
  EXPECT_TRUE(plan.predecessors(read_task).empty());
  EXPECT_TRUE(plan.successors(read_task).empty());
}

TEST(PhysicalQepTransformationTest, FilterAppendsFilterTaskAndTerminalGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::filter_relation filter{read, {}, less_condition(0, 1), identity_indices(2)};

  auto const plan = qep::physical_qep_transformation(&filter, &provider);

  // The filter emits a masked table, so a terminal gather materializes the final result.
  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  EXPECT_EQ(plan.predecessors(filter_task), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{filter_task});
  EXPECT_TRUE(plan.successors(gather_task).empty());
}

TEST(PhysicalQepTransformationTest, NonIdentityFilterProjectionStaysMaskedUntilTerminal)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::filter_relation filter{read, {}, less_condition(0, 1), {1}};

  auto const plan = qep::physical_qep_transformation(&filter, &provider);

  // read -> filter -> project; the project passes the valid mask through, so the terminal
  // still needs a gather.
  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* filter_task  = single_task_of_type<qep::filter_task>(plan);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  auto const* gather_task  = single_task_of_type<qep::gather_task>(plan);
  ASSERT_NE(project_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  EXPECT_EQ(plan.predecessors(project_task), std::vector<qep::task const*>{filter_task});
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{project_task});
}

TEST(PhysicalQepTransformationTest, ProjectOnPlainInputNeedsNoGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));
  physical::project_relation project{read, {}, std::move(exprs)};

  auto const plan = qep::physical_qep_transformation(&project, &provider);

  ASSERT_EQ(plan.tasks().size(), 2);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  ASSERT_NE(project_task, nullptr);
  EXPECT_TRUE(plan.successors(project_task).empty());
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, IdentityProjectIsElided)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.push_back(col_ref(0));
  exprs.push_back(col_ref(1));
  physical::project_relation project{read, {}, std::move(exprs)};

  auto const plan = qep::physical_qep_transformation(&project, &provider);

  // An identity projection is a no-op; the read lowers as the plan's only task.
  ASSERT_EQ(plan.tasks().size(), 1);
  EXPECT_NE(single_task_of_type<storage::oneshot_parquet_read_task>(plan), nullptr);
}

TEST(PhysicalQepTransformationTest, ReorderingProjectIsNotElided)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.push_back(col_ref(1));
  exprs.push_back(col_ref(0));
  physical::project_relation project{read, {}, std::move(exprs)};

  auto const plan = qep::physical_qep_transformation(&project, &provider);

  ASSERT_EQ(plan.tasks().size(), 2);
  EXPECT_NE(single_task_of_type<qep::project_task>(plan), nullptr);
}

TEST(PhysicalQepTransformationTest, KeylessAggregateLowersToReduceAggregate)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  values.emplace_back(cudf::aggregation::COUNT_ALL, col_ref(0));
  physical::concatenate_aggregate_relation aggregate{read, {}, /*keys=*/{}, std::move(values)};

  auto const plan = qep::physical_qep_transformation(&aggregate, &provider);

  ASSERT_EQ(plan.tasks().size(), 2);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* reduce_task = single_task_of_type<qep::reduce_aggregate_task>(plan);
  ASSERT_NE(reduce_task, nullptr);
  EXPECT_EQ(plan.predecessors(reduce_task), std::vector<qep::task const*>{read_task});
  EXPECT_TRUE(plan.successors(reduce_task).empty());
}

TEST(PhysicalQepTransformationTest, AggregateConditionLowersToUpstreamFilter)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, /*keys=*/{}, std::move(values), less_condition(0, 1)};

  auto const plan = qep::physical_qep_transformation(&aggregate, &provider);

  // The reduce aggregate consumes the filter's masked output natively — no gather in between,
  // and the aggregate's plain output needs no terminal gather either.
  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
  auto const* reduce_task = single_task_of_type<qep::reduce_aggregate_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(reduce_task, nullptr);
  EXPECT_EQ(plan.predecessors(reduce_task), std::vector<qep::task const*>{filter_task});
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, ComputedAggregateValueThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM,
                      std::make_unique<gqe::multiply_expression>(
                        std::make_shared<gqe::column_reference_expression>(0),
                        std::make_shared<gqe::column_reference_expression>(1)));
  physical::concatenate_aggregate_relation aggregate{read, {}, /*keys=*/{}, std::move(values)};

  EXPECT_THROW(std::ignore = qep::physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, PerfectHashGroupByLowersAndCombinesCountPartialsWithSum)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::COUNT_VALID, col_ref(1));
  values.emplace_back(cudf::aggregation::COUNT_ALL, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  auto const plan = qep::physical_qep_transformation(&aggregate, &provider);

  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* read_task    = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* partial_base = single_task_of_type<qep::perfect_hash_partial_aggregate_task>(plan);
  auto const* combine_base = single_task_of_type<qep::perfect_hash_aggregate_task>(plan);
  ASSERT_NE(read_task, nullptr);
  ASSERT_NE(partial_base, nullptr);
  ASSERT_NE(combine_base, nullptr);
  EXPECT_EQ(plan.predecessors(partial_base), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(combine_base), std::vector<qep::task const*>{partial_base});
  EXPECT_TRUE(plan.successors(combine_base).empty());

  auto const* partial = dynamic_cast<qep::perfect_hash_partial_aggregate_task const*>(partial_base);
  auto const* combine = dynamic_cast<qep::perfect_hash_aggregate_task const*>(combine_base);
  ASSERT_NE(partial, nullptr);
  ASSERT_NE(combine, nullptr);

  auto chunk1 =
    make_table(int64_column_wrapper{1, 1}.release(), int64_column_wrapper{10, 11}.release());
  auto chunk2 =
    make_table(int64_column_wrapper{1, 1, 1}.release(), int64_column_wrapper{12, 13, 14}.release());
  gqe::context_reference ctx_ref{};
  auto partial1 = partial->next(chunk1, ctx_ref);
  auto partial2 = partial->next(chunk2, ctx_ref);
  ASSERT_TRUE(partial1.has_value());
  ASSERT_TRUE(partial2.has_value());

  auto accumulator = combine->initialize(ctx_ref);
  combine->next(*partial1, accumulator, ctx_ref);
  combine->next(*partial2, accumulator, ctx_ref);
  auto result      = combine->finalize(std::move(accumulator), ctx_ref);
  auto result_view = qep::to_table_view(result);
  ASSERT_TRUE(result_view.has_value());

  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{1}.release());
  expected_columns.push_back(int64_column_wrapper{5}.release());
  expected_columns.push_back(int64_column_wrapper{5}.release());
  auto expected = cudf::table{std::move(expected_columns)};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result_view, expected.view());
}

TEST(PhysicalQepTransformationTest, PerfectHashAggregateConditionDoesNotInsertGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{read,
                                                     {},
                                                     std::move(keys),
                                                     std::move(values),
                                                     less_condition(0, 1),
                                                     /*perfect_hashing=*/true};

  auto const plan = qep::physical_qep_transformation(&aggregate, &provider);

  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* filter_task  = single_task_of_type<qep::filter_task>(plan);
  auto const* partial_task = single_task_of_type<qep::perfect_hash_partial_aggregate_task>(plan);
  auto const* combine_task = single_task_of_type<qep::perfect_hash_aggregate_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(partial_task, nullptr);
  ASSERT_NE(combine_task, nullptr);
  EXPECT_EQ(plan.predecessors(partial_task), std::vector<qep::task const*>{filter_task});
  EXPECT_EQ(plan.predecessors(combine_task), std::vector<qep::task const*>{partial_task});
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, NonPerfectHashGroupByRemainsUnsupported)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{read, {}, std::move(keys), std::move(values)};

  EXPECT_THROW(std::ignore = qep::physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, PerfectHashComputedKeyThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  EXPECT_THROW(std::ignore = qep::physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, PerfectHashComputedValueThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM,
                      std::make_unique<gqe::multiply_expression>(
                        std::make_shared<gqe::column_reference_expression>(0),
                        std::make_shared<gqe::column_reference_expression>(1)));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  EXPECT_THROW(std::ignore = qep::physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, SubqueryInFilterConditionThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 1, nullptr);
  auto haystack       = make_read("t", 1, nullptr);

  std::vector<std::shared_ptr<gqe::expression>> needles;
  needles.push_back(std::make_shared<gqe::column_reference_expression>(0));
  auto condition = std::make_unique<gqe::in_predicate_expression>(std::move(needles), 0);
  physical::filter_relation filter{read, {haystack}, std::move(condition), identity_indices(1)};

  EXPECT_THROW(std::ignore = qep::physical_qep_transformation(&filter, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, UnsupportedRelationKindThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 1, nullptr);
  physical::fetch_relation fetch{read, /*offset=*/0, /*count=*/10};

  EXPECT_THROW(std::ignore = qep::physical_qep_transformation(&fetch, &provider), std::logic_error);
}

TEST(PhysicalQepTransformationTest, InPredicatePartialFilterWiresHaystackPredecessor)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto haystack       = make_read("t", 1, nullptr);

  std::vector<std::shared_ptr<gqe::expression>> needles;
  needles.push_back(std::make_shared<gqe::column_reference_expression>(0));
  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(std::move(needles), 0);
  auto read           = make_read("t", 1, std::move(partial_filter), {haystack});

  auto const plan = qep::physical_qep_transformation(read.get(), &provider);

  // Two reads: the haystack subquery and the pruned main read, wired haystack -> read.
  auto const reads = tasks_of_type<storage::oneshot_parquet_read_task>(plan);
  ASSERT_EQ(plan.tasks().size(), 2);
  ASSERT_EQ(reads.size(), 2);
  auto const* pruned = plan.predecessors(reads[0]).empty() ? reads[1] : reads[0];
  ASSERT_EQ(plan.predecessors(pruned).size(), 1);
  EXPECT_TRUE(plan.successors(pruned).empty());
}

TEST(PhysicalQepTransformationTest, MaskedHaystackIsMaterializedBeforePrunedRead)
{
  auto const provider  = make_provider_with_parquet_table("t");
  auto haystack_source = make_read("t", 1, nullptr);
  auto haystack =
    std::make_shared<physical::filter_relation>(haystack_source,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 0),
                                                identity_indices(1));

  std::vector<std::shared_ptr<gqe::expression>> needles;
  needles.push_back(std::make_shared<gqe::column_reference_expression>(0));
  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(std::move(needles), 0);
  auto read           = make_read("t", 1, std::move(partial_filter), {haystack});

  auto const plan = qep::physical_qep_transformation(read.get(), &provider);

  // haystack read -> filter -> gather -> pruned read: the pruned read rejects masked input.
  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_EQ(plan.successors(gather_task).size(), 1);
  EXPECT_NE(
    dynamic_cast<storage::oneshot_parquet_read_task const*>(plan.successors(gather_task).front()),
    nullptr);
}

TEST(PhysicalQepTransformationTest, WriteAppendsFoldTask)
{
  auto provider = make_provider_with_parquet_table("t");
  provider.add_table(
    "out", std::make_shared<storage::parquet_table>(std::vector<std::string>{"out.parquet"}));
  auto read = make_read("t", 2, nullptr);
  physical::write_relation write{read, {"col_0", "col_1"}, "out"};

  auto const plan = qep::physical_qep_transformation(&write, &provider);

  ASSERT_EQ(plan.tasks().size(), 2);
  auto const* read_task = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* fold_task = single_task_of_type<storage::chunked_parquet_write_task>(plan);
  ASSERT_NE(fold_task, nullptr);
  EXPECT_EQ(plan.predecessors(fold_task), std::vector<qep::task const*>{read_task});
  EXPECT_TRUE(plan.successors(fold_task).empty());
}

TEST(PhysicalQepTransformationTest, MaskedInputToWriteIsMaterialized)
{
  auto provider = make_provider_with_parquet_table("t");
  provider.add_table(
    "out", std::make_shared<storage::parquet_table>(std::vector<std::string>{"out.parquet"}));
  auto read = make_read("t", 2, nullptr);
  auto filter =
    std::make_shared<physical::filter_relation>(read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(2));
  physical::write_relation write{filter, {"col_0", "col_1"}, "out"};

  auto const plan = qep::physical_qep_transformation(&write, &provider);

  // read -> filter -> gather -> write: the writer fold rejects masked input.
  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  auto const* fold_task   = single_task_of_type<storage::chunked_parquet_write_task>(plan);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_NE(fold_task, nullptr);
  EXPECT_EQ(plan.predecessors(fold_task), std::vector<qep::task const*>{gather_task});
}
