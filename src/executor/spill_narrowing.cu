/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, flowcsd study. SPDX-License-Identifier: Apache-2.0
 *
 * flowcsd SSD spill mode: lossless narrowing of spilled integer columns
 * (include/gqe/executor/spill_narrowing.hpp).
 */

#include <gqe/executor/spill_narrowing.hpp>

#include <gqe/utility/error.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace gqe {

namespace {

// cudf::contiguous_split and cudf::pack pad every column buffer of a packed table to this many bytes
// (cudf contiguous_split.cu split_align).
constexpr std::size_t pack_buffer_alignment_bytes = 64;

constexpr int threads_per_block = 256;
constexpr int64_t max_grid_size = 16384;  // the kernel loops over larger columns (grid-stride)

// The integer type a column's values are counted in when narrowing applies to it, else nullopt.
// Integers as they are. DATE (TIMESTAMP_DAYS, an INT32 count of days) and the other timestamps
// (INT64 counts) narrow exactly like integers; floating point, strings, decimals and the rest stay.
std::optional<cudf::data_type> integer_representation(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: return type;
    case cudf::type_id::TIMESTAMP_DAYS: return cudf::data_type{cudf::type_id::INT32};
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return cudf::data_type{cudf::type_id::INT64};
    default: return std::nullopt;
  }
}

// A valid integer scalar's value as 64 bits (two's complement), so largest - smallest is exact.
struct integer_scalar_bits {
  template <typename T>
  uint64_t operator()(cudf::scalar const& value, rmm::cuda_stream_view stream) const
  {
    if constexpr (cudf::is_integral_not_bool<T>()) {
      auto const& integer = static_cast<cudf::numeric_scalar<T> const&>(value);
      return static_cast<uint64_t>(integer.value(stream));
    } else {
      throw std::logic_error("spill: narrowing expects an integer scalar");
    }
  }
};

// The smallest unsigned type that holds 0 .. range.
cudf::data_type narrowest_unsigned_type(uint64_t range)
{
  if (range <= std::numeric_limits<uint8_t>::max()) {
    return cudf::data_type{cudf::type_id::UINT8};
  }
  if (range <= std::numeric_limits<uint16_t>::max()) {
    return cudf::data_type{cudf::type_id::UINT16};
  }
  if (range <= std::numeric_limits<uint32_t>::max()) {
    return cudf::data_type{cudf::type_id::UINT32};
  }
  return cudf::data_type{cudf::type_id::UINT64};
}

std::size_t padded_pack_buffer_bytes(std::size_t bytes)
{
  return (bytes + pack_buffer_alignment_bytes - 1) / pack_buffer_alignment_bytes *
         pack_buffer_alignment_bytes;
}

// One slice's stored values of the column being restored.
struct restore_piece {
  uint8_t const* data;   // stored values
  uint64_t min_bits;     // added back; 0 when the piece was stored as is
  int32_t stored_bytes;  // 1, 2, 4 or 8
};

// output[row] = min + stored value of the piece holding `row` (the last piece starting at or before
// it); the sum's low bits are exactly the original value's bits, signed or not.
template <typename Word>
__global__ void restore_pieces(restore_piece const* pieces,
                               int64_t const* piece_starts,
                               int32_t num_pieces,
                               Word* output,
                               int64_t num_rows)
{
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < num_rows;
       row += stride) {
    int32_t low  = 0;
    int32_t high = num_pieces - 1;
    while (low < high) {
      int32_t const middle = (low + high + 1) / 2;
      if (piece_starts[middle] <= row) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    restore_piece const piece = pieces[low];
    int64_t const index       = row - piece_starts[low];
    uint64_t stored           = 0;
    switch (piece.stored_bytes) {
      case 1: stored = piece.data[index]; break;
      case 2: stored = reinterpret_cast<uint16_t const*>(piece.data)[index]; break;
      case 4: stored = reinterpret_cast<uint32_t const*>(piece.data)[index]; break;
      default: stored = reinterpret_cast<uint64_t const*>(piece.data)[index]; break;
    }
    output[row] = static_cast<Word>(piece.min_bits + stored);
  }
}

template <typename Word>
void launch_restore(rmm::device_uvector<restore_piece> const& pieces,
                    rmm::device_uvector<int64_t> const& piece_starts,
                    rmm::device_buffer& output,
                    int64_t num_rows,
                    rmm::cuda_stream_view stream)
{
  auto const blocks = std::clamp<int64_t>(
    (num_rows + threads_per_block - 1) / threads_per_block, 1, max_grid_size);
  restore_pieces<Word><<<static_cast<int>(blocks), threads_per_block, 0, stream.value()>>>(
    pieces.data(),
    piece_starts.data(),
    static_cast<int32_t>(pieces.size()),
    static_cast<Word*>(output.data()),
    num_rows);
  GQE_CUDA_TRY(cudaGetLastError());
}

}  // namespace

narrowed_rows narrow_rows(cudf::table_view rows, rmm::cuda_stream_view stream)
{
  // Smallest and largest value of every narrowable column, all queued before any is read.
  std::vector<std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>>> ranges(
    rows.num_columns());
  for (cudf::size_type index = 0; index < rows.num_columns(); index++) {
    auto const& column        = rows.column(index);
    auto const representation = integer_representation(column.type());
    if (!representation || column.null_count() == column.size()) { continue; }
    ranges[index] = cudf::minmax(cudf::bit_cast(column, *representation), stream);
  }

  auto plan = std::make_shared<narrowing_plan>();
  narrowed_rows narrowed;
  std::vector<cudf::column_view> columns;
  for (cudf::size_type index = 0; index < rows.num_columns(); index++) {
    auto const& column = rows.column(index);
    auto& entry        = plan->emplace_back(column_narrowing{column.type(), column.type(), 0});
    auto const& [smallest, largest] = ranges[index];
    if (smallest == nullptr) {
      columns.push_back(column);
      continue;
    }
    auto const representation = smallest->type();
    auto const smallest_bits =
      cudf::type_dispatcher(representation, integer_scalar_bits{}, *smallest, stream);
    auto const largest_bits =
      cudf::type_dispatcher(representation, integer_scalar_bits{}, *largest, stream);
    auto const stored_type = narrowest_unsigned_type(largest_bits - smallest_bits);
    if (cudf::size_of(stored_type) >= cudf::size_of(representation)) {
      columns.push_back(column);
      continue;
    }
    // value - smallest fits the stored type exactly: every value lies in [smallest, largest].
    narrowed.columns.push_back(cudf::binary_operation(cudf::bit_cast(column, representation),
                                                      *smallest,
                                                      cudf::binary_operator::SUB,
                                                      stored_type,
                                                      stream));
    columns.push_back(narrowed.columns.back()->view());
    entry.stored_type = stored_type;
    entry.min_bits    = smallest_bits;
  }
  narrowed.view = cudf::table_view(columns);
  if (!narrowed.columns.empty()) { narrowed.plan = std::move(plan); }
  return narrowed;
}

std::size_t unnarrowed_packed_bytes(std::size_t packed_bytes,
                                    cudf::size_type rows,
                                    narrowing_plan const* plan)
{
  if (plan == nullptr) { return packed_bytes; }
  auto const num_rows = static_cast<std::size_t>(rows);
  for (auto const& entry : *plan) {
    if (!entry.narrowed()) { continue; }
    packed_bytes += padded_pack_buffer_bytes(num_rows * cudf::size_of(entry.original_type)) -
                    padded_pack_buffer_bytes(num_rows * cudf::size_of(entry.stored_type));
  }
  return packed_bytes;
}

std::unique_ptr<cudf::column> restore_column(std::vector<cudf::column_view> const& pieces,
                                             std::vector<column_narrowing> const& narrowing,
                                             rmm::cuda_stream_view stream)
{
  // Where each piece's stored values are, where they go in the output, and what to add back.
  std::vector<restore_piece> host_pieces;
  std::vector<int64_t> host_starts;
  int64_t num_rows           = 0;
  cudf::size_type null_count = 0;
  for (std::size_t piece = 0; piece < pieces.size(); piece++) {
    auto const& stored      = pieces[piece];
    auto const stored_bytes = cudf::size_of(stored.type());
    auto const* first_value =
      stored.head<uint8_t>() + static_cast<std::size_t>(stored.offset()) * stored_bytes;
    host_pieces.push_back(
      restore_piece{first_value, narrowing[piece].min_bits, static_cast<int32_t>(stored_bytes)});
    host_starts.push_back(num_rows);
    num_rows += stored.size();
    null_count += stored.null_count();
  }
  GQE_EXPECTS(num_rows <= std::numeric_limits<cudf::size_type>::max(),
              "spill: a restored partition column exceeds the cudf column size limit");

  // From pageable host memory: the host vectors may go as soon as the copies are queued.
  rmm::device_uvector<restore_piece> device_pieces(host_pieces.size(), stream);
  rmm::device_uvector<int64_t> device_starts(host_starts.size(), stream);
  GQE_CUDA_TRY(cudaMemcpyAsync(device_pieces.data(),
                               host_pieces.data(),
                               host_pieces.size() * sizeof(restore_piece),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  GQE_CUDA_TRY(cudaMemcpyAsync(device_starts.data(),
                               host_starts.data(),
                               host_starts.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  // The original type's integer bits; DATE and TIMESTAMP columns hold their integer count.
  auto const original_type = narrowing.front().original_type;
  auto const value_bytes   = cudf::size_of(original_type);
  rmm::device_buffer data(static_cast<std::size_t>(num_rows) * value_bytes, stream);
  switch (value_bytes) {
    case 1: launch_restore<uint8_t>(device_pieces, device_starts, data, num_rows, stream); break;
    case 2: launch_restore<uint16_t>(device_pieces, device_starts, data, num_rows, stream); break;
    case 4: launch_restore<uint32_t>(device_pieces, device_starts, data, num_rows, stream); break;
    case 8: launch_restore<uint64_t>(device_pieces, device_starts, data, num_rows, stream); break;
    default: throw std::logic_error("spill: no narrowing for a column of this width");
  }
  rmm::device_buffer null_mask;
  if (null_count > 0) { null_mask = cudf::concatenate_masks(pieces, stream); }
  return std::make_unique<cudf::column>(original_type,
                                        static_cast<cudf::size_type>(num_rows),
                                        std::move(data),
                                        std::move(null_mask),
                                        null_count);
}

}  // namespace gqe
