/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, flowcsd study. SPDX-License-Identifier: Apache-2.0
 *
 * flowcsd SSD spill mode: the probe-side filter of a hybrid hash join (see spill.hpp).
 *
 * A spilled inner join (either build side) or left semi join (build on the right) drops every probe
 * row without a matching build key. The task-graph builder gives such a join a filter only when its
 * build side is filtered (otherwise TPC-H's foreign keys find a build key for every probe key). The
 * filter is built over all build keys in stage A, before the build rows are partitioned, so the
 * keys of spilled build partitions are in it too, and it is applied in stage B before the probe
 * rows are partitioned: a probe row that cannot match is neither joined with the resident build nor
 * written to SSD.
 *
 * Kind, decided at the first build task with rows:
 *   bitmap  exact, one bit per key value. Used when the key is one integer column and the estimated
 *           key span of all build tasks (the first task's span x number of build tasks: tables are
 *           stored in key order) needs at most `bitmap_max_bits_per_row` bits per estimated build
 *           row (the build keys are dense) and fits `max_filter_bytes` of bits. The bits live in
 *           1 MiB pages allocated as keys arrive, so a key range that grows task by task (a table
 *           scanned in key order) is never copied. If the key range grows past `max_filter_bytes`
 *           of bits, the keys move into a Bloom filter.
 *   bloom   otherwise: a chain of blocked Bloom filters (cuco::bloom_filter_ref: blocks of 8 x 32-bit
 *           words, each key sets one bit per word) with `bloom_bits_per_row` bits per key a filter
 *           is sized for, about 1 % false positives. The first is sized for the estimated build
 *           rows. When a build task's keys would take the newest filter past the keys it was sized
 *           for, they go to a new filter sized for twice that (or for the task, if more), so an
 *           estimate that comes out low (a first build task smaller than the others, e.g. a hybrid
 *           join's stage-B output) costs memory, not filtering. A probe key passes when any filter
 *           of the chain may hold it. The filters hold the integer key value, or for other keys
 *           (several columns, strings, ...) the cudf xxhash_64 of the key columns.
 * Estimated build rows = rows of the first build task with rows x number of build tasks.
 * Neither kind grows past `max_filter_bytes` (a first Bloom filter is cut to it). When the next
 * filter would take a chain past it, the join stops filtering: the chain is freed and every probe row
 * passes (kind none), which is always correct. Null keys are never added; a probe row with a null key
 * is kept (the join decides).
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gqe {

/**
 * @brief Probe-side filter of one hybrid join (see the file comment). Not thread-safe: the spill set
 * calls it under its mutex.
 */
class probe_filter {
 public:
  enum class kind { none, bitmap, bloom };

  /**
   * @param[in] build_tasks Number of stage-A tasks of the build side; sizes the Bloom filter.
   */
  explicit probe_filter(int64_t build_tasks);

  probe_filter(probe_filter const&)            = delete;
  probe_filter& operator=(probe_filter const&) = delete;

  /**
   * @brief Stage A: add the keys of one build task (one column per equality key).
   */
  void add_build_keys(std::vector<cudf::column_view> const& keys, rmm::cuda_stream_view stream);

  /**
   * @brief Stage B: the probe rows whose keys may match a build key, or nullptr when that is every
   * row (no copy). `keys` are the probe rows' key columns.
   */
  std::unique_ptr<cudf::table> filter_probe_rows(cudf::table_view rows,
                                                 std::vector<cudf::column_view> const& keys,
                                                 rmm::cuda_stream_view stream);

  /**
   * @brief Free the filter's device memory (stage B is over); kind and false-positive estimate stay.
   */
  void release();

  [[nodiscard]] kind filter_kind() const noexcept { return _kind; }

  /**
   * @brief Bloom filters in the chain (0 for a bitmap); still counted after release().
   */
  [[nodiscard]] std::size_t bloom_filters() const noexcept { return _bloom_chain.size(); }

  /**
   * @brief Whether the join stopped filtering because its Bloom chain would pass max_filter_bytes.
   */
  [[nodiscard]] bool stopped() const noexcept { return _stopped; }

  /**
   * @brief Build rows added so far.
   */
  [[nodiscard]] int64_t build_rows() const noexcept { return _build_rows; }

  /**
   * @brief Device memory the filter holds now (bitmap pages and page table, or the blocks of the
   * Bloom chain).
   */
  [[nodiscard]] std::size_t device_bytes() const noexcept;

  /**
   * @brief Expected share of non-matching probe keys the filter lets through: 0 for a bitmap; for a
   * Bloom chain 1 - prod(1 - rate of each filter), each from the blocked-Bloom model for the keys
   * added to it (duplicates counted, so an upper estimate when build keys repeat).
   */
  [[nodiscard]] double false_positive_rate() const;

 private:
  // One Bloom filter of the chain. release() frees its blocks; its counts stay for the
  // false-positive estimate.
  struct chained_bloom_filter {
    rmm::device_buffer blocks;
    std::size_t num_blocks = 0;
    int64_t capacity_keys  = 0;  // keys it was sized for
    int64_t keys           = 0;  // keys added (build rows, or distinct keys moved from a bitmap)
  };

  // Bitmap: add one build task's keys (INT64 values with their key range), adding pages as needed.
  // False, with nothing changed, when the key range would grow past max_filter_bytes of bits.
  bool add_to_bitmap(cudf::column_view values,
                     int64_t smallest_key,
                     int64_t largest_key,
                     rmm::cuda_stream_view stream);
  // Bloom: start the chain with a filter for `expected_keys` keys (cut to max_filter_bytes).
  void start_bloom(int64_t expected_keys, rmm::cuda_stream_view stream);
  // Bloom: append a filter for `capacity_keys` keys; false, with filtering stopped, when it would
  // take the chain past max_filter_bytes.
  bool grow_bloom(int64_t capacity_keys, rmm::cuda_stream_view stream);
  // Bloom: append a zeroed filter of `num_blocks` blocks sized for `capacity_keys` keys.
  void append_bloom(std::size_t num_blocks, int64_t capacity_keys, rmm::cuda_stream_view stream);
  // Bloom: add one build task's 64-bit values (integer key values or row hashes) to the newest
  // filter, or to a new one when they would take the newest past the keys it was sized for.
  void add_to_bloom(uint64_t const* values, int64_t count, rmm::cuda_stream_view stream);
  // The key range grew past max_filter_bytes: move the bitmap's keys into a Bloom filter.
  void switch_to_bloom(int64_t expected_keys, rmm::cuda_stream_view stream);
  // Rebuild the device page table after the page range or the pages changed.
  void upload_page_table(rmm::cuda_stream_view stream);

  int64_t _build_tasks;
  kind _kind                    = kind::none;
  bool _stopped                 = false;  // the Bloom chain would pass max_filter_bytes: no filter
  bool _bloom_holds_hashes      = false;  // bloom: xxhash_64 of the key columns, else the key value
  int64_t _estimated_build_rows = 0;
  int64_t _build_rows           = 0;  // build rows added so far

  // Bitmap: page i covers key values (first_page + i) * page bits + bit; empty = no key there yet.
  int64_t _first_page = 0;
  std::vector<rmm::device_buffer> _pages;
  std::vector<uint64_t*> _page_pointers;  // host copy of the page table (source of its upload)
  rmm::device_buffer _page_table;

  // Bloom filters, oldest first.
  std::vector<chained_bloom_filter> _bloom_chain;
};

}  // namespace gqe
