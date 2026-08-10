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

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/perfect_hash_aggregate.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/operator/reduce_aggregate.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <cassert>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

/**
 * @brief Map an aggregation to the aggregation that combines its partial results.
 *
 * Associative reductions combine with the same operation; counts combine by summing the partial
 * counts.
 */
[[nodiscard]] cudf::aggregation::Kind combine_aggregation_kind(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::SUM:
    case cudf::aggregation::PRODUCT:
    case cudf::aggregation::MIN: return kind;
    case cudf::aggregation::COUNT_VALID:
    case cudf::aggregation::COUNT_ALL: return cudf::aggregation::SUM;
    default:
      throw std::invalid_argument(
        "physical_qep_transformation: unsupported aggregation kind for perfect-hash aggregate");
  }
}

/**
 * @brief Derive the combine stage's aggregate inputs from the partial aggregate specification.
 *
 * The partial task emits `[keys..., partial aggregates...]`, so combine value `i` reads column
 * `num_keys + i`.
 */
[[nodiscard]] std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>
derive_combine_value_indices(
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> const& value_indices,
  cudf::size_type num_keys)
{
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> combine_value_indices;
  combine_value_indices.reserve(value_indices.size());
  for (std::size_t aggregate_idx = 0; aggregate_idx < value_indices.size(); ++aggregate_idx) {
    combine_value_indices.emplace_back(combine_aggregation_kind(value_indices[aggregate_idx].first),
                                       num_keys + static_cast<cudf::size_type>(aggregate_idx));
  }
  return combine_value_indices;
}

/**
 * @brief The terminal QEP task lowered from a single physical relation.
 *
 * Records the terminal so the parent's lowering can wire it as a predecessor, plus whether the
 * terminal emits a masked-table state container (so parents can decide whether to insert a
 * `gather_task` for downstream consumers that require a regular-table shape).
 */
struct lowering_result {
  task* terminal = nullptr;  ///< Terminal QEP task of the lowered relation.
  bool is_masked = false;    ///< `true` if the terminal emits `[valid_mask, columns...]`.
};

/**
 * @brief `true` when `indices` is `[0, 1, ..., input_column_count - 1]`.
 */
[[nodiscard]] bool is_identity_projection(std::vector<cudf::size_type> const& indices,
                                          std::size_t input_column_count)
{
  assert(indices.size() <= std::numeric_limits<cudf::size_type>::max());

  if (indices.size() != input_column_count) return false;
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(indices.size()); ++i) {
    if (indices[i] != i) return false;
  }
  return true;
}

/**
 * @brief `true` when `exprs` is `[col_ref(0), col_ref(1), ..., col_ref(input_column_count - 1)]`.
 */
[[nodiscard]] bool is_identity_projection(std::vector<expression*> const& exprs,
                                          std::size_t input_column_count)
{
  assert(exprs.size() <= std::numeric_limits<cudf::size_type>::max());

  if (exprs.size() != input_column_count) return false;
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(exprs.size()); ++i) {
    if (exprs[i] == nullptr || exprs[i]->type() != expression::expression_type::column_reference) {
      return false;
    }
    if (static_cast<column_reference_expression const*>(exprs[i])->column_idx() != i) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Throw a uniformly-shaped error indicating that a physical relation kind is not yet
 *        supported in QEP lowering.
 */
[[noreturn]] void throw_unsupported(std::string_view kind)
{
  throw std::logic_error(std::format(
    "physical_qep_transformation: physical relation kind not yet supported in QEP lowering: {}",
    kind));
}

/**
 * @brief Inject a `gather_task` ahead of `upstream` when it produces a masked-table state
 *        container.
 *
 * Used at points where the downstream consumer requires a regular-table shape — currently
 * the haystack input of a partition-pruned read, the writer fold, and the QEP terminal.
 *
 * @param[in] qep_builder The QEP builder collecting tasks and edges.
 * @param[in] upstream The upstream lowering result.
 *
 * @return The terminal QEP task that downstream consumers should use as their predecessor.
 */
task* materialize_if_masked(query_execution_plan_builder* qep_builder, lowering_result upstream)
{
  if (!upstream.is_masked) { return upstream.terminal; }

  auto gather      = std::make_unique<gather_task>();
  auto* gather_ptr = gather.get();

  qep_builder->add_task(std::move(gather));
  qep_builder->add_successor(upstream.terminal, gather_ptr);

  return gather_ptr;
}

/**
 * @brief Internal visitor that lowers a physical relation tree to QEP tasks.
 *
 * The visitor works as follows:
 *
 * 1. `lower(rel)` probes the cache. If hit, then it returns the cached entry. On a miss, it
 * invokes the visitor and caches the published result.
 * 2. Each `visit(...)` overload lowers the relation passed to it and publishes its terminal in
 * `_result` — the return channel from the void-returning `accept`.
 *
 * `lower` is the only cache writer. The cache deduplicates a relation pointer encountered
 * twice, so each physical relation is lowered exactly once. Every recursive descent into
 * children goes through `lower` (not raw `accept`), so visits never need to consult the cache
 * themselves.
 */
class qep_lowering_visitor : public physical::relation_visitor {
 public:
  /**
   * @brief Construct a lowering visitor.
   *
   * @param[in] provider Table provider used to resolve `read_relation::table_name`. Must
   *   outlive the visitor.
   * @param[in] qep_builder The QEP builder collecting tasks and edges. Must outlive the
   *   visitor.
   */
  qep_lowering_visitor(storage::table_provider const* provider,
                       query_execution_plan_builder* qep_builder)
    : _table_provider(provider), _qep_builder(qep_builder)
  {
  }

  /**
   * @brief Lower `rel`, caching the result by relation pointer to share subplans.
   *
   * @param[in] rel The physical relation to lower.
   *
   * @return The terminal QEP task and its masked-output flag.
   */
  lowering_result lower(physical::relation* rel)
  {
    if (auto it = _cache.find(rel); it != _cache.end()) { return it->second; }

    _result = {};
    rel->accept(*this);
    GQE_EXPECTS(_result.terminal != nullptr,
                "physical_qep_transformation: visit(...) did not publish a lowering result");

    _cache[rel] = _result;
    return _result;
  }

  void visit(physical::read_relation* rel) override
  {
    auto const table_name = rel->table_name();
    auto tbl              = _table_provider->get_table(table_name);
    if (!tbl || !tbl->is_readable()) {
      throw std::logic_error(std::format(
        "physical_qep_transformation: read_relation for table `{}` has no readable storage",
        table_name));
    }

    // `partial_filter` is a predicate-pushdown hint. Semantically, exact filtering remains the
    // downstream `filter_relation`'s responsibility.
    //
    // Lower the partial filter as follows:
    //
    //  - Decode an in-predicate subquery on a single column reference into a partition-column
    //  pruner.
    //  - Add the subquery as a predecessor edge.
    //  - Pass-through the filter verbatim for the read task to use as it sees fit.
    std::optional<cudf::size_type> partition_column_idx;
    lowering_result haystack_result;
    auto* partial_filter = rel->partial_filter_unsafe();
    if (partial_filter != nullptr &&
        partial_filter->type() == expression::expression_type::subquery) {
      auto const* sq = static_cast<subquery_expression const*>(partial_filter);
      if (sq->subquery_type() == subquery_expression::subquery_type_type::in_predicate) {
        auto const needles = sq->children();
        GQE_EXPECTS(
          needles.size() == 1,
          "physical_qep_transformation: in-predicate partial filter must have exactly one needle "
          "expression");
        GQE_EXPECTS(needles[0]->type() == expression::expression_type::column_reference,
                    "physical_qep_transformation: in-predicate partial filter needle must be a "
                    "column reference");
        auto const* col_ref  = static_cast<column_reference_expression const*>(needles[0]);
        partition_column_idx = col_ref->column_idx();

        auto const subqueries = rel->subqueries_unsafe();
        GQE_EXPECTS(
          sq->relation_index() < subqueries.size(),
          "physical_qep_transformation: in-predicate subquery relation_index out of range");
        haystack_result = lower(subqueries[sq->relation_index()]);
      }
    }

    auto iterate = tbl->readable_view()->get_iterate_task(
      rel->column_names(),
      rel->output_data_types(),
      partition_column_idx,
      partial_filter != nullptr ? partial_filter->clone() : nullptr);
    auto* read_ptr = iterate.get();

    // Reads have no input dataflow by default. Pruned reads receive their haystack as the
    // single predecessor edge.
    _qep_builder->add_task(std::move(iterate));

    if (partition_column_idx) {
      // The read task consumes its haystack via `qep::to_table_view`, which rejects masks —
      // materialize upstream if needed.
      auto* haystack_ptr = materialize_if_masked(_qep_builder, haystack_result);
      _qep_builder->add_successor(haystack_ptr, read_ptr);
    }

    _result = {read_ptr, /*is_masked=*/false};
  }

  void visit(physical::filter_relation* rel) override
  {
    auto* condition = rel->condition_unsafe();
    GQE_EXPECTS(!contains_subquery(condition),
                "physical_qep_transformation: filter conditions containing subquery_expression "
                "are not supported (unnest scalar subqueries into joins upstream)");

    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: filter_relation must have exactly one child");
    auto const child = lower(children[0]);

    auto const filter_result = lower_into_filter(child.terminal, condition->clone());

    // Insert a project task to compute non-identity expressions. The `filter_task` does not apply
    // `projection_indices`. As the project passes the filter's valid mask through, the result stays
    // masked.
    auto const& projection_indices = rel->projection_indices();
    auto const child_column_count  = children[0]->output_data_types().size();
    if (is_identity_projection(projection_indices, child_column_count)) {
      _result = filter_result;
      return;
    }

    _result = lower_into_project(filter_result, projection_indices);
  }

  void visit(physical::project_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: project_relation must have exactly one child");
    auto const child = lower(children[0]);

    // Elide identity projections.
    //
    // Besides being no-ops, projections are pipelined operators and thus cannot be applied to
    // materialized edges (e.g. re-emitting an aggregate's output).
    auto const output_expressions = rel->output_expressions_unsafe();
    if (is_identity_projection(output_expressions, children[0]->output_data_types().size())) {
      _result = child;
      return;
    }

    _result = lower_into_project(child, clone_expressions(output_expressions));
  }

  void visit(physical::concatenate_aggregate_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(
      children.size() == 1,
      "physical_qep_transformation: concatenate_aggregate_relation must have exactly one child");
    auto child = lower(children[0]);

    // `aggregate_relation_base::condition` is technical debt inherited from the task graph design.
    // It's a pre-aggregation WHERE-style filter applied to the *input* (see
    // `physical/aggregate.hpp`).
    //
    // The QEP unnests the filter as a
    // `filter_task` ahead of the aggregate. `reduce_aggregate_task` consumes the resulting
    // masked-table natively, so no gather is inserted between them.
    auto* condition = rel->condition_unsafe();
    if (condition != nullptr) {
      GQE_EXPECTS(!contains_subquery(condition),
                  "physical_qep_transformation: aggregate input conditions containing "
                  "subquery_expression are not supported (unnest scalar subqueries into joins "
                  "upstream)");
      child = lower_into_filter(child.terminal, condition->clone());
    }

    // Lower scalar aggregate.
    if (rel->keys_unsafe().empty()) {
      // Flatten each aggregation function expression into a (function kind, column index) pair.
      //
      // The `complex_expression_extraction_into_project` optimizer rule simplifies complex
      // expressions such that we see only a bare column reference here (except `COUNT_ALL`, which
      // reads no column).
      auto const values = rel->values_unsafe();
      std::vector<std::pair<cudf::aggregation::Kind, std::optional<cudf::size_type>>> value_indices;
      value_indices.reserve(values.size());
      for (auto const& [kind, expr] : values) {
        if (kind == cudf::aggregation::COUNT_ALL) {
          value_indices.emplace_back(kind, std::nullopt);
        } else {
          GQE_EXPECTS(
            expr != nullptr && expr->type() == expression::expression_type::column_reference,
            "physical_qep_transformation: aggregate value expressions must be bare column "
            "references "
            "(run the `complex_expression_extraction_into_project` optimizer rule to materialize "
            "computed measures upstream)");
          value_indices.emplace_back(
            kind, static_cast<column_reference_expression const*>(expr)->column_idx());
        }
      }

      auto reduce_task =
        std::make_unique<reduce_aggregate_task>(std::move(value_indices), rel->output_data_types());
      auto* reduce_ptr = reduce_task.get();
      _qep_builder->add_task(std::move(reduce_task));
      _qep_builder->add_successor(child.terminal, reduce_ptr);

      _result = {reduce_ptr, /*is_masked=*/false};
    } else {
      if (!rel->perfect_hashing()) {
        throw_unsupported("group-by aggregate without perfect hashing");
      }

      // The default optimizer materializes complex aggregate expressions upstream. Perfect-hash
      // tasks therefore reference the remaining bare column references directly.
      auto const keys = rel->keys_unsafe();
      std::vector<cudf::size_type> key_indices;
      key_indices.reserve(keys.size());
      for (auto const* key : keys) {
        GQE_EXPECTS(
          key != nullptr && key->type() == expression::expression_type::column_reference,
          "physical_qep_transformation: aggregate key expressions must be bare column references "
          "(run the `complex_expression_extraction_into_project` optimizer rule to materialize "
          "computed grouping keys upstream)");
        key_indices.push_back(static_cast<column_reference_expression const*>(key)->column_idx());
      }

      auto const values = rel->values_unsafe();
      std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices;
      value_indices.reserve(values.size());
      for (auto const& [kind, value] : values) {
        GQE_EXPECTS(
          value != nullptr && value->type() == expression::expression_type::column_reference,
          "physical_qep_transformation: aggregate value expressions must be bare column "
          "references "
          "(run the `complex_expression_extraction_into_project` optimizer rule to materialize "
          "computed measures upstream)");
        value_indices.emplace_back(
          kind, static_cast<column_reference_expression const*>(value)->column_idx());
      }

      auto const num_keys        = static_cast<cudf::size_type>(key_indices.size());
      auto combine_value_indices = derive_combine_value_indices(value_indices, num_keys);
      std::vector<cudf::size_type> combine_key_indices(key_indices.size());
      for (cudf::size_type key_idx = 0; key_idx < num_keys; ++key_idx) {
        combine_key_indices[key_idx] = key_idx;
      }

      auto partial_task = std::make_unique<perfect_hash_partial_aggregate_task>(
        std::move(key_indices), std::move(value_indices), rel->output_data_types());
      auto* partial_ptr = partial_task.get();
      _qep_builder->add_task(std::move(partial_task));
      _qep_builder->add_successor(child.terminal, partial_ptr);

      auto combine_task = std::make_unique<perfect_hash_aggregate_task>(
        std::move(combine_key_indices), std::move(combine_value_indices), rel->output_data_types());
      auto* combine_ptr = combine_task.get();
      _qep_builder->add_task(std::move(combine_task));
      _qep_builder->add_successor(partial_ptr, combine_ptr);

      _result = {combine_ptr, /*is_masked=*/false};
    }
  }

  void visit(physical::write_relation* rel) override
  {
    auto const table_name = rel->table_name();
    auto tbl              = _table_provider->get_table(table_name);
    if (!tbl || !tbl->is_writeable()) {
      throw std::logic_error(std::format(
        "physical_qep_transformation: write_relation for table `{}` has no writeable storage",
        table_name));
    }

    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: write_relation must have exactly one child");
    auto const child = lower(children[0]);

    // The writer fold consumes its input via `qep::to_table_view`, which rejects masks —
    // materialize if the upstream emits a masked-table.
    auto* upstream_ptr = materialize_if_masked(_qep_builder, child);

    auto fold = tbl->writeable_view()->get_fold_task(rel->column_names(), rel->output_data_types());
    auto* fold_ptr = fold.get();
    _qep_builder->add_task(std::move(fold));
    _qep_builder->add_successor(upstream_ptr, fold_ptr);

    _result = {fold_ptr, /*is_masked=*/false};
  }

  void visit(physical::broadcast_join_relation*) override
  {
    throw_unsupported("broadcast_join (pending the QEP hash-join operators)");
  }
  void visit(physical::shuffle_join_relation*) override { throw_unsupported("shuffle_join"); }
  void visit(physical::concatenate_sort_relation*) override
  {
    throw_unsupported("concatenate_sort (pending a QEP sort operator)");
  }
  void visit(physical::fetch_relation*) override { throw_unsupported("fetch"); }
  void visit(physical::union_all_relation*) override { throw_unsupported("union_all"); }
  void visit(physical::user_defined_relation*) override { throw_unsupported("user_defined"); }
  void visit(physical::window_relation*) override { throw_unsupported("window"); }
  void visit(physical::gen_ident_col_relation*) override { throw_unsupported("gen_ident_col"); }
  void visit(physical::shuffle_relation*) override { throw_unsupported("shuffle"); }

 private:
  /**
   * @brief Append a `filter_task` evaluating `condition` on top of `upstream`.
   *
   * `filter_task` accepts both regular and masked-table inputs, so filter chains compose
   * directly without an intervening `gather_task`.
   *
   * @param[in] upstream The upstream terminal task feeding the filter.
   * @param[in] condition Owned filter condition expression.
   *
   * @return The terminal `filter_task`; its output is always a masked table.
   */
  lowering_result lower_into_filter(task* upstream, std::unique_ptr<expression> condition)
  {
    auto filter      = std::make_unique<filter_task>(std::move(condition));
    auto* filter_ptr = filter.get();

    _qep_builder->add_task(std::move(filter));
    _qep_builder->add_successor(upstream, filter_ptr);

    return {filter_ptr, /*is_masked=*/true};
  }

  /**
   * @brief Append a `project_task` evaluating `exprs` on top of `upstream`.
   *
   * `project_task` accepts both regular and masked-table inputs (see `project.hpp`), so no
   * gather is inserted.
   *
   * @param[in] upstream The upstream lowering result feeding the project.
   * @param[in] exprs Owned output expressions for the project task.
   *
   * @return The terminal `project_task`; an input valid mask is passed through to the output,
   *   so the output shape matches the input's.
   */
  lowering_result lower_into_project(lowering_result upstream,
                                     std::vector<std::unique_ptr<expression>> exprs)
  {
    auto project      = std::make_unique<project_task>(std::move(exprs));
    auto* project_ptr = project.get();

    _qep_builder->add_task(std::move(project));
    _qep_builder->add_successor(upstream.terminal, project_ptr);

    return {project_ptr, upstream.is_masked};
  }

  /**
   * @brief Append a `project_task` that selects a subset of upstream columns by index.
   *
   * Synthesizes a `column_reference_expression` per index and delegates to the
   * expression-based `lower_into_project`. Many physical relations carry an index-based
   * column remap (e.g. `filter_relation::projection_indices`) that maps to this pattern.
   *
   * @param[in] upstream The upstream lowering result feeding the project.
   * @param[in] column_indices Column indices into the upstream's column layout, in output
   *   order.
   *
   * @return The terminal `project_task`; an input valid mask is passed through to the output,
   *   so the output shape matches the input's.
   */
  lowering_result lower_into_project(lowering_result upstream,
                                     std::vector<cudf::size_type> const& column_indices)
  {
    std::vector<std::unique_ptr<expression>> exprs;
    exprs.reserve(column_indices.size());

    for (auto idx : column_indices) {
      exprs.push_back(std::make_unique<column_reference_expression>(idx));
    }

    return lower_into_project(upstream, std::move(exprs));
  }

  storage::table_provider const* _table_provider;  ///< Non-owning. Resolves table names to
                                                   ///< storage tables for `read_relation`.
  query_execution_plan_builder* _qep_builder;      ///< Non-owning. Collects lowered tasks and
                                                   ///< their dataflow edges.
  std::unordered_map<physical::relation*, lowering_result>
    _cache;  ///< Subplan deduplication: each `physical::relation*` is lowered at most once;
             ///< repeated occurrences reuse the cached result. Written only by `lower`.
  lowering_result _result;  ///< Result published by the most recent `visit(...)`; consumed by
                            ///< `lower` immediately after `accept` returns.
};

}  // namespace

query_execution_plan physical_qep_transformation(physical::relation* root,
                                                 storage::table_provider const* table_provider)
{
  GQE_EXPECTS(
    root != nullptr, "physical_qep_transformation: root must not be null", std::invalid_argument);
  GQE_EXPECTS(table_provider != nullptr,
              "physical_qep_transformation: table_provider must not be null",
              std::invalid_argument);

  query_execution_plan_builder qep_bld;

  qep_lowering_visitor visitor{table_provider, &qep_bld};
  auto const root_result = visitor.lower(root);

  // The QEP terminal must produce a regular-table state container so `result()` callers can
  // read it through `qep::to_table_view`. Inject a final `gather_task` if the terminal emits
  // a masked-table.
  materialize_if_masked(&qep_bld, root_result);

  return qep_bld.build();
}

}  // namespace qep
}  // namespace gqe
