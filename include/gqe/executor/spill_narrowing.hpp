/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, flowcsd study. SPDX-License-Identifier: Apache-2.0
 *
 * flowcsd SSD spill mode: lossless narrowing of the integer columns of spilled rows (see spill.hpp).
 *
 * A column of the rows being written is narrowed when its values are integers (INT8 .. UINT64) or
 * DATE/TIMESTAMP values, whose integer count of days (seconds, ...) narrows exactly like an
 * integer, and (largest - smallest) fits a narrower unsigned type: it is stored as
 * (value - smallest) in the smallest of UINT8, UINT16 and UINT32 that holds it. Floating-point,
 * string, decimal and other columns are stored as they are. Reading adds smallest back.
 * The spill set narrows the rows of one input task once (narrow_rows) and writes slices of the
 * result; a spilled partition gathers slices of many tasks, each with its own smallest value, and
 * restore_column rebuilds one of its columns from all of them in one GPU pass.
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gqe {

/**
 * @brief How one column of spilled rows is stored.
 */
struct column_narrowing {
  cudf::data_type original_type;
  cudf::data_type stored_type;  // UINT8/16/32 when narrowed, else original_type
  uint64_t min_bits = 0;        // value subtracted (two's complement bits of its integer count)

  [[nodiscard]] bool narrowed() const noexcept { return stored_type != original_type; }
};

/**
 * @brief One entry per column of the spilled rows.
 */
using narrowing_plan = std::vector<column_narrowing>;

/**
 * @brief Rows as written: the narrowed columns are new columns, the others the rows' own.
 */
struct narrowed_rows {
  std::shared_ptr<narrowing_plan const> plan;  // null: no column got narrower (`view` = the rows)
  std::vector<std::unique_ptr<cudf::column>> columns;  // the narrowed columns, backing `view`
  cudf::table_view view;
};

/**
 * @brief Narrow the integer columns of `rows`: one cudf::minmax per such column and one
 * subtraction per column that gets narrower.
 */
narrowed_rows narrow_rows(cudf::table_view rows, rmm::cuda_stream_view stream);

/**
 * @brief The packed size a slice of `rows` rows would have had without narrowing, from its packed
 * size: cudf::contiguous_split and cudf::pack pad every buffer to 64 bytes, so only the narrowed
 * data buffers differ.
 */
std::size_t unnarrowed_packed_bytes(std::size_t packed_bytes,
                                    cudf::size_type rows,
                                    narrowing_plan const* plan);

/**
 * @brief One column of a spilled partition, in its original type, from the stored columns of its
 * slices in order; `narrowing[i]` says how `pieces[i]` was stored. Null masks are concatenated.
 */
std::unique_ptr<cudf::column> restore_column(std::vector<cudf::column_view> const& pieces,
                                             std::vector<column_narrowing> const& narrowing,
                                             rmm::cuda_stream_view stream);

}  // namespace gqe
