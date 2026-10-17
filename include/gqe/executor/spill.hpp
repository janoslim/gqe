/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, flowcsd study. SPDX-License-Identifier: Apache-2.0
 *
 * flowcsd SSD spill mode (study gpu-storage-baselines, PLAN §9).
 *
 * A hybrid hash join or a hybrid keyed aggregation keeps in GPU memory as much of the input it
 * would otherwise hold whole (a join's build side, an aggregation's partial results) as fits a
 * budget, and sends the rest through the data SSDs. How much spills is decided at run time from the
 * actual bytes; an operator whose held input fits spills nothing. Each operator has its own
 * partition count P (spill_partitions_for).
 *   stage A : spill_partition_task per build (or partial-result) task: hash-partition on the keys.
 *             Partitions 0 .. k-1 stay in GPU memory. The first held task with rows projects k from
 *             its bytes times the number of held tasks; before a task's resident partitions are
 *             copied, room is made for them. The operators of a query share one resident budget:
 *             whenever their resident bytes (and probe-side filters) would exceed it, the operator
 *             that still can (its stage B or C has not started) and holds the most resident bytes
 *             writes out its highest resident partition and its k drops. An operator whose held
 *             side is smaller than one 8 MiB write staging buffer keeps every partition and is
 *             never chosen (unless the budget is 0). The task's partitions >= k are packed in one
 *             batch and written to SSD (cuFile/GDS) by background I/O threads while the next input
 *             is scanned and partitioned. A join that drops probe rows
 *             without a match (inner, or left semi with the build on the right) and whose build
 *             side is filtered also adds every build key to its probe-side filter first
 *             (spill_filter.hpp).
 *   stage B : (joins) spill_probe_split_task per probe task: the probe-side filter drops rows
 *             that cannot match; of the rest, rows of partitions < k join the resident build at
 *             once (spill_resident_build_task; one hash table for all probe tasks), rows of
 *             partitions >= k are written to SSD. With k = P the probe rows are neither filtered
 *             nor partitioned: the join runs like the broadcast join.
 *   stage C : spill_partition_done_task per partition runs the partition's pipeline only when it
 *             has spill data on some side (or, partition 0 of an aggregation, resident data):
 *             spill_read_task per side, then the unchanged join_task / aggregate_task. A spilled
 *             partition is read back from SSD (the next two are read ahead while this one is
 *             processed). A join's resident partitions were joined in stage B; an aggregation's
 *             resident partitions all come back as partition 0 (one final aggregation for them).
 * Written slices store only the columns the join reads (dead columns get a placeholder when read
 * back) and store integer columns, and DATE/TIMESTAMP columns through their integer count, as
 * (value - min) in the smallest unsigned type that holds their input task's range; the read adds
 * min back (lossless narrowing, spill_narrowing.hpp).
 * The task-graph builder (task_graph.cpp) inserts these tasks; nothing else changes.
 *
 * Settings come from the environment (read once per process; the task manager inherits the node
 * manager's environment):
 *   GQE_SPILL_PARTITIONS             default P, for operators without an entry below; 0 or unset =
 *                                    spill mode off, and the task graph is the upstream one.
 *   GQE_SPILL_OPERATOR_PARTITIONS    P per operator: "<signature>=<P>;<signature>=<P>;..." (the
 *                                    signature is the builder's SF-independent operator name, see
 *                                    the `flowcsd_spill_plan` lines; it contains '=' itself, so each
 *                                    entry splits at its last '=').
 *   GQE_SPILL_RESIDENT_BUDGET_BYTES  held bytes the operators of a query may keep in GPU memory
 *                                    together; unset = 1/4 of the query pool (the correctness check
 *                                    sets it small to force spilling at a small scale factor; 0
 *                                    spills every partition of every operator; the query pool keeps
 *                                    every partition resident).
 *   GQE_SPILL_DIRS                   ':'-separated spill directories, one per drive.
 */

#pragma once

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/utility/uuid.hpp>

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gqe {

/**
 * @brief Spill-mode settings from the environment (see the file comment).
 */
struct spill_settings {
  int32_t partitions = 0;                               // default P; > 0 turns spill mode on
  std::map<std::string, int32_t> operator_partitions;  // signature -> P
  std::optional<std::size_t> resident_budget_bytes;
  std::vector<std::string> directories;
};

/**
 * @brief Return the spill settings, read from the environment on the first call.
 */
spill_settings const& get_spill_settings();

/**
 * @brief An operator's partition count P and where it came from.
 */
struct operator_partitions {
  int32_t partitions;
  bool from_table;  // true: GQE_SPILL_OPERATOR_PARTITIONS; false: the default GQE_SPILL_PARTITIONS
};

/**
 * @brief P of the operator with this signature: its GQE_SPILL_OPERATOR_PARTITIONS entry, else the
 * default GQE_SPILL_PARTITIONS.
 */
operator_partitions spill_partitions_for(std::string const& signature);

/**
 * @brief Held bytes the operators of a query may keep in GPU memory together:
 * GQE_SPILL_RESIDENT_BUDGET_BYTES, else 1/4 of the query pool.
 *
 * With 1/4, stage B also fits one operator's resident build twice (slices and their concatenation)
 * and its hash table next to the probe tasks, and stage C fits two spilled partitions (each at
 * most 1/8 of the pool) with the next ones read ahead.
 */
std::size_t spill_resident_budget_bytes(std::size_t pool_bytes);

/**
 * @brief Kinds of operators the task-graph builder puts on the spill path.
 */
enum class hybrid_operator { join, aggregation };

/**
 * @brief Count one operator on the spill path for the per-query `flowcsd_spill` log line.
 */
void count_hybrid_operator(hybrid_operator kind);

/**
 * @brief Reset the spill counters; called when a query starts (one query runs at a time).
 */
void reset_spill_counters();

/**
 * @brief Log the spill counters of the query: one `flowcsd_spill_drive` line per spill directory
 * and one `flowcsd_spill task_manager ...` line.
 */
void log_spill_counters(int rank, utility::uuid const& query_id);

/**
 * @brief Registered GPU staging buffers and background I/O threads of the running query (defined
 * in spill.cpp).
 */
class spill_io;

/**
 * @brief The spill I/O of the running query: created by the first call, shared while anyone holds
 * it. plan_executor holds it for the whole query in spill mode, so the operators of a query share
 * one set of buffers and threads.
 */
std::shared_ptr<spill_io> acquire_spill_io();

/**
 * @brief How the output rows of a stage-C join tell how many of its spilled probe rows matched.
 */
enum class spill_match_rule {
  output_rows,              // inner join with unique build keys, semi join: matched = output rows
  probe_rows_minus_output,  // anti join: matched = probe rows - output rows
  unknown,                  // left join, inner join with duplicate build keys, aggregation
};

/**
 * @brief What the task-graph builder decided about one operator on the spill path.
 */
struct spill_operator_plan {
  // 2 for a join (side 0 = left input, 1 = right input), 1 for an aggregation.
  int32_t num_sides = 1;
  std::string description;   // what the operator is (the `flowcsd_spill_plan` line of the builder)
  std::string signature;     // SF-independent name of the operator (task_graph.cpp)
  int32_t partitions = 0;    // P
  int32_t held_side  = -1;   // side whose partitions may stay resident; -1: every partition to SSD
  int32_t probe_side = -1;   // a join's probe side, -1 for an aggregation
  int64_t held_input_tasks = 0;          // stage-A tasks of the held side (projects k)
  int64_t probe_filter_build_tasks = 0;  // > 0: the join gets a probe-side filter for this many
                                         // build tasks (spill_filter.hpp); 0: no filter
  // Per side: input columns the join reads neither in its condition nor in its projection. Written
  // slices leave them out (fixed-width and string columns; others are stored anyway).
  std::array<std::vector<cudf::size_type>, 2> dead_columns;
  std::size_t resident_budget_bytes = 0;  // spill_resident_budget_bytes of the query
  std::size_t pool_bytes            = 0;  // query pool: limits a partition to 1/8 of it
};

/**
 * @brief Host-side owner of the partitions of one operator on the spill path (defined in
 * spill.cpp).
 *
 * Per side it keeps one append file per drive; per (side, partition) the resident slices (GPU
 * tables) or the index of its written slices (drive, 4 KiB-aligned file offset, packed and restored
 * size, cuDF pack metadata, narrowing). A join that drops unmatched probe rows may also own its
 * probe-side filter. It deletes a side's files once its spilled partitions have been read back, or
 * when the operator is destroyed, and then logs one `flowcsd_spill_operator` line per side.
 */
class spill_set;

/**
 * @brief Create the spill set of one operator.
 */
std::shared_ptr<spill_set> make_spill_set(spill_operator_plan plan);

/**
 * @brief Stage A: hash-partition one input table on its keys; keep resident slices in GPU memory
 * while they fit and hand the others to the background writers.
 *
 * The result is an empty table; the partitions come back through `spill_read_task` and, for a
 * join's resident build partitions, `spill_resident_build_task`.
 */
class spill_partition_task : public task {
 public:
  /**
   * @param[in] input Input table to be partitioned.
   * @param[in] key_exprs Key expressions evaluated on `input`.
   * @param[in] key_column_offset Offset of the column references in `key_exprs`, as in
   * `evaluate_expressions` (a join's right keys reference the concatenation of both inputs).
   * @param[in] spill Spill set of the operator.
   * @param[in] side Input side within the operator.
   */
  spill_partition_task(context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<task> input,
                       std::vector<std::unique_ptr<expression>> key_exprs,
                       cudf::size_type key_column_offset,
                       std::shared_ptr<spill_set> spill,
                       int32_t side);

  void execute() override;

 private:
  std::vector<std::unique_ptr<expression>> _key_exprs;
  cudf::size_type _key_column_offset;
  std::shared_ptr<spill_set> _spill;
  int32_t _side;
};

/**
 * @brief Stage B: split one probe table; drop the rows the probe-side filter rules out, write the
 * rows of spilled partitions to SSD and return the rows of resident partitions for the join with
 * the resident build.
 *
 * The result is borrowed: from the partitioned copy this task keeps, or, when every partition is
 * resident, from the input task, which this task then keeps alive.
 */
class spill_probe_split_task : public task {
 public:
  /**
   * @param[in] input Probe table to be split; the other parameters as in `spill_partition_task`.
   */
  spill_probe_split_task(context_reference ctx_ref,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<task> input,
                         std::vector<std::unique_ptr<expression>> key_exprs,
                         cudf::size_type key_column_offset,
                         std::shared_ptr<spill_set> spill,
                         int32_t side);

  void execute() override;

 private:
  std::shared_ptr<task> _input;  // backs the borrowed result when the probe rows pass through
  std::vector<std::unique_ptr<expression>> _key_exprs;
  cudf::size_type _key_column_offset;
  std::shared_ptr<spill_set> _spill;
  int32_t _side;
  std::unique_ptr<cudf::table> _partitioned;  // backs the borrowed result otherwise
};

/**
 * @brief Stage B: the resident build partitions of a join as one table (the broadcast table of the
 * stage-B joins). Its memory leaves the query's resident budget when this task is destroyed.
 */
class spill_resident_build_task : public task {
 public:
  /**
   * @param[in] partition_tasks All stage-A `spill_partition_task`s of the build side.
   */
  spill_resident_build_task(context_reference ctx_ref,
                            int32_t task_id,
                            int32_t stage_id,
                            std::vector<std::shared_ptr<task>> partition_tasks,
                            std::shared_ptr<spill_set> spill,
                            int32_t side);
  ~spill_resident_build_task() override;

  void execute() override;

 private:
  std::shared_ptr<spill_set> _spill;
  int32_t _side;
};

/**
 * @brief Stage C: one partition of one side as a single table: its slices read back from SSD, or
 * (partition 0) the side's resident partitions.
 */
class spill_read_task : public task {
 public:
  /**
   * @param[in] partition_tasks Stage-A `spill_partition_task`s of this side (none for a probe side
   * split in stage B). They ran in an earlier stage; depending on them keeps them alive until every
   * partition has been read.
   */
  spill_read_task(context_reference ctx_ref,
                  int32_t task_id,
                  int32_t stage_id,
                  std::vector<std::shared_ptr<task>> partition_tasks,
                  std::shared_ptr<spill_set> spill,
                  int32_t side,
                  int32_t partition);
  ~spill_read_task() override;

  void execute() override;

 private:
  std::shared_ptr<spill_set> _spill;
  int32_t _side;
  int32_t _partition;
};

/**
 * @brief Stage C: the root of one partition's pipeline. Runs the partition's join (or final
 * aggregation) only when the partition has spill data on some side, or resident data (partition 0
 * of an aggregation); otherwise it emits an empty table of the operator's output types without
 * running its dependencies. Afterwards it records the partition's stage-C time (its first spill
 * read to this point) and, for a join, how many of its spilled probe rows matched.
 *
 * The result borrows the joined (or aggregated) rows; this task keeps their task alive.
 */
class spill_partition_done_task : public task {
 public:
  /**
   * @param[in] output_types Output column types of the operator (its empty result).
   */
  spill_partition_done_task(context_reference ctx_ref,
                            int32_t task_id,
                            int32_t stage_id,
                            std::shared_ptr<task> input,
                            std::shared_ptr<spill_set> spill,
                            int32_t partition,
                            spill_match_rule match_rule,
                            std::vector<cudf::data_type> output_types);

  void execute() override;

 private:
  std::shared_ptr<task> _input;  // backs the borrowed result
  std::shared_ptr<spill_set> _spill;
  int32_t _partition;
  spill_match_rule _match_rule;
  std::vector<cudf::data_type> _output_types;
};

}  // namespace gqe
