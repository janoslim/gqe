/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

/* IMPORTANT NOTE: Any changes to the in-class members  will break serialization and
   deserialization. Both need to be updated.*/

#include <gqe/executor/aggregate.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace gqe {
namespace storage {

bool compressed_sliced_buffer_view::is_compressed() const
{
  return _compression_format != gqe::compression_format::none;
}

gqe::compression_format compressed_sliced_buffer_view::compression_format() const
{
  return _compression_format;
}

cudf::data_type compressed_sliced_buffer_view::element_type() const { return _element_type; }

partition_span compressed_sliced_buffer_view::get_partition(size_t partition_idx) const
{
  if (partition_idx >= _buffers.size()) {
    throw std::out_of_range("compressed_sliced_buffer_view partition index out of range");
  }
  auto const* data = reinterpret_cast<uint8_t const*>(_buffers[partition_idx].data());
  return partition_span{data, _buffers[partition_idx].size()};
}

template <class Allocator>
compressed_sliced_column_storage<Allocator>::compressed_sliced_column_storage(
  cudf::column&& cudf_column,
  int partition_size,
  memory_kind::type memory_kind,
  gqe::compression::configuration compression_config,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  allocator_type const& alloc)
  : compressed_sliced_column_storage(
      cudf_column, partition_size, std::move(compression_config), alloc)
{
  compress(std::move(cudf_column), memory_kind, stream, mr);
}

template <class Allocator>
compressed_sliced_column_storage<Allocator>::compressed_sliced_column_storage(
  const cudf::column& cudf_column,
  int partition_size,
  gqe::compression::configuration compression_config,
  allocator_type const& alloc)
  : _cudf_type(cudf_column.type()),
    _null_counts(gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc)),
    _compression_config(std::move(compression_config)),
    _data_buffer(cudf_column.type(), alloc),
    _null_mask_buffer(cudf::data_type(cudf::type_id::INT32), alloc)
{
  if (partition_size <= 0) {
    throw std::invalid_argument("compressed_sliced_column requires a positive partition size.");
  }
  _partition_size = static_cast<size_t>(partition_size);
  _size           = cudf_column.size();
}

template <class Allocator>
compressed_sliced_column_storage<Allocator>
compressed_sliced_column_storage<Allocator>::deserialize_from_disk(
  std::filesystem::path const& file_path,
  std::string const& column_name,
  size_t row_group_index,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  allocator_type const& alloc)
{
  (void)column_name;
  (void)row_group_index;
  (void)stream;
  (void)mr;
  (void)alloc;
  GQE_LOG_TRACE("deserialization not implemented yet (path='{}')", file_path.string());
  throw std::runtime_error("Deserialize not implemented");
}

template <class Allocator>
compressed_sliced_column_storage<Allocator>::compressed_sliced_column_storage(
  size_t size,
  size_t partition_size,
  cudf::data_type cudf_type,
  gqe::compression::configuration compression_config,
  bool is_compressed,
  bool is_null_mask_compressed,
  bool is_secondary_compressed,
  std::vector<cudf::size_type>&& compressed_data_sizes,
  std::vector<cudf::size_type>&& compressed_null_mask_sizes,
  std::vector<cudf::size_type>&& null_counts,
  std::vector<rmm::device_buffer>&& compressed_data_buffers,
  std::vector<rmm::device_buffer>&& compressed_null_masks,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  allocator_type const& alloc)
  : _size(size),
    _partition_size(partition_size),
    _cudf_type(cudf_type),
    _null_counts(null_counts.begin(),
                 null_counts.end(),
                 gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc)),
    _compression_config(std::move(compression_config)),
    _data_buffer(cudf_type, alloc),
    _null_mask_buffer(cudf::data_type(cudf::type_id::INT32), alloc)
{
  (void)compressed_data_sizes;
  (void)compressed_null_mask_sizes;

  if (partition_size == 0) {
    throw std::invalid_argument("compressed_sliced_column deserialize: partition_size is zero");
  }

  _data_buffer.buffers.reserve(compressed_data_buffers.size());
  for (auto const& compressed_buffer : compressed_data_buffers) {
    _data_buffer.buffers.emplace_back(compressed_buffer, stream, mr);
  }
  _data_buffer.compression_format =
    is_compressed ? (is_secondary_compressed ? _compression_config.secondary_compression_format
                                             : _compression_config.primary_compression_format)
                  : gqe::compression_format::none;
  _data_buffer.compressed_size =
    std::accumulate(_data_buffer.buffers.begin(),
                    _data_buffer.buffers.end(),
                    size_t{0},
                    [](size_t total, auto const& buffer) { return total + buffer.size(); });
  _data_buffer.uncompressed_size =
    size == 0 ? 0 : static_cast<size_t>(size) * cudf::size_of(cudf_type);
  if (is_secondary_compressed) {
    _data_buffer.primary_compressed_size   = 0;
    _data_buffer.secondary_compressed_size = _data_buffer.compressed_size;
  } else if (is_compressed) {
    _data_buffer.primary_compressed_size   = _data_buffer.compressed_size;
    _data_buffer.secondary_compressed_size = 0;
  } else {
    _data_buffer.primary_compressed_size   = 0;
    _data_buffer.secondary_compressed_size = 0;
  }

  _null_mask_buffer.buffers.reserve(compressed_null_masks.size());
  for (auto const& compressed_null_mask : compressed_null_masks) {
    _null_mask_buffer.buffers.emplace_back(compressed_null_mask, stream, mr);
  }
  _null_mask_buffer.compression_format =
    is_null_mask_compressed
      ? (is_secondary_compressed ? _compression_config.secondary_compression_format
                                 : _compression_config.primary_compression_format)
      : gqe::compression_format::none;
  _null_mask_buffer.compressed_size =
    std::accumulate(_null_mask_buffer.buffers.begin(),
                    _null_mask_buffer.buffers.end(),
                    size_t{0},
                    [](size_t total, auto const& buffer) { return total + buffer.size(); });
  if (_null_mask_buffer.buffers.empty()) {
    _null_mask_buffer.uncompressed_size = 0;
  } else {
    size_t const num_partitions = gqe::utility::divide_round_up(size, partition_size);
    size_t const tail_rows      = size - (num_partitions - 1) * partition_size;
    size_t const full_words     = gqe::utility::divide_round_up(partition_size, size_t{32});
    size_t const tail_words     = gqe::utility::divide_round_up(tail_rows, size_t{32});
    _null_mask_buffer.uncompressed_size =
      ((num_partitions - 1) * full_words + tail_words) * sizeof(cudf::bitmask_type);
  }
  if (is_null_mask_compressed) {
    _null_mask_buffer.primary_compressed_size   = _null_mask_buffer.compressed_size;
    _null_mask_buffer.secondary_compressed_size = 0;
  } else {
    _null_mask_buffer.primary_compressed_size   = 0;
    _null_mask_buffer.secondary_compressed_size = 0;
  }
}

template <class Allocator>
void compressed_sliced_column_storage<Allocator>::do_compress(
  rmm::device_buffer const* input,
  typename compressed_sliced_column_storage<Allocator>::buffer_storage& output,
  size_t num_rows,
  size_t num_partitions,
  bool is_null_mask,
  memory_kind::type memory_kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  utility::nvtx_scoped_range compress_batch_range("compress_batch");

  const uint8_t* bytes_buffer = reinterpret_cast<const uint8_t*>(input->data());
  size_t start_ix             = 0;
  auto const data_type = gqe::compression::get_optimal_nvcomp_data_type(output.element_type.id());

  std::vector<rmm::device_buffer> device_uncompressed_data_buffers;
  std::vector<const void*> input_ptrs;
  std::vector<size_t> input_sizes;
  std::vector<void*> device_ptrs;
  device_uncompressed_data_buffers.reserve(num_partitions);
  input_ptrs.reserve(num_partitions);
  input_sizes.reserve(num_partitions);
  device_ptrs.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t row_start_ix = ix_partition * _partition_size;
    size_t row_count    = std::min(num_rows - row_start_ix, _partition_size);
    size_t size         = row_count * cudf::size_of(output.element_type);
    if (is_null_mask) { size = gqe::utility::divide_round_up(row_count, 32) * 4; }

    device_uncompressed_data_buffers.emplace_back(size, stream, mr);
    input_ptrs.push_back(reinterpret_cast<const void*>(bytes_buffer + start_ix));
    input_sizes.push_back(size);
    device_ptrs.push_back(reinterpret_cast<void*>(device_uncompressed_data_buffers.back().data()));
    start_ix += size;
  }

  gqe::utility::do_batched_memcpy((void**)device_ptrs.data(),
                                  (void**)input_ptrs.data(),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream);

  bool try_secondary_compression = is_null_mask ? false : true;
  std::vector<cudf::size_type> compressed_sizes;
  auto compressed_buffers =
    gqe::compression::compress_batch(_compression_config,
                                     data_type,
                                     std::move(device_uncompressed_data_buffers),
                                     output.compression_format,
                                     output.compressed_size,
                                     output.uncompressed_size,
                                     output.primary_compressed_size,
                                     output.secondary_compressed_size,
                                     compressed_sizes,
                                     memory_kind,
                                     try_secondary_compression,
                                     stream,
                                     mr);
  // rmm::device_buffer does not expose ownership transfer for its allocation, so persist each
  // compressed partition with an explicit copy into gqe::storage::buffer storage.
  output.buffers.reserve(compressed_buffers.size());
  for (auto const& compressed_buffer : compressed_buffers) {
    output.buffers.emplace_back(compressed_buffer, stream, mr);
  }

  stream.synchronize();
}

template <class Allocator>
void compressed_sliced_column_storage<Allocator>::compress(cudf::column&& cudf_column,
                                                           memory_kind::type memory_kind,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  const size_t num_rows        = cudf_column.size();
  const size_t num_partitions  = gqe::utility::divide_round_up(num_rows, _partition_size);
  const auto column_null_count = cudf_column.null_count();
  // Setting up offsets for the cudf slice API

  auto column_content = cudf_column.release();
  assert(column_content.children.empty());

  do_compress(column_content.data.get(),
              _data_buffer,
              num_rows,
              num_partitions,
              false /*is_null_mask*/,
              memory_kind,
              stream,
              mr);

  if (column_null_count > 0) {
    do_compress(column_content.null_mask.get(),
                _null_mask_buffer,
                num_rows,
                num_partitions,
                true /*is_null_mask*/,
                memory_kind,
                stream,
                mr);
    rmm::device_buffer null_mask_buffer(
      column_content.null_mask->data(), column_content.null_mask->size(), stream, mr);
    _null_counts.reserve(num_partitions);
    for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
      // Get the null count for this partition
      size_t start_ix = ix_partition * _partition_size;
      size_t stop_ix  = std::min(start_ix + _partition_size, num_rows);
      _null_counts.push_back(cudf::null_count(
        reinterpret_cast<const uint32_t*>(null_mask_buffer.data()), start_ix, stop_ix, stream));
    }
  }
}

template <class Allocator>
void compressed_sliced_column_storage<Allocator>::deallocate(rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  // TODO: Fix system_memory_resource, pinned_memory_resource, numa_memory_resource, and
  // boost_shared_memory_resource, which immediately free host/shared memory while ignoring the
  // stream. Remove this once they satisfy RMM stream-ordered semantics.
  stream.synchronize();
  _data_buffer.deallocate(stream, mr);
  _null_mask_buffer.deallocate(stream, mr);
}

template <class Allocator>
int64_t compressed_sliced_column_storage<Allocator>::size() const
{
  return _size;
}

template <class Allocator>
cudf::size_type compressed_sliced_column_storage<Allocator>::null_count() const
{
  return std::accumulate(_null_counts.begin(), _null_counts.end(), cudf::size_type{0});
}

template <class Allocator>
bool compressed_sliced_column_storage<Allocator>::is_compressed() const
{
  return _data_buffer.compression_format != gqe::compression_format::none;
}

template <class Allocator>
int64_t compressed_sliced_column_storage<Allocator>::get_compressed_size() const
{
  return static_cast<int64_t>(_data_buffer.compressed_size + _null_mask_buffer.compressed_size);
}

template <class Allocator>
int64_t compressed_sliced_column_storage<Allocator>::get_uncompressed_size() const
{
  return static_cast<int64_t>(_data_buffer.uncompressed_size + _null_mask_buffer.uncompressed_size);
}

template <class Allocator>
column_compression_statistics compressed_sliced_column_storage<Allocator>::get_compression_stats()
  const
{
  fixed_width_compression_statistics fixed_width_stats;
  fixed_width_stats.compressed_size   = get_compressed_size();
  fixed_width_stats.uncompressed_size = get_uncompressed_size();

  fixed_width_stats.primary_compressed_size   = _data_buffer.primary_compressed_size;
  fixed_width_stats.secondary_compressed_size = _data_buffer.secondary_compressed_size;
  auto const primary_format                   = _compression_config.primary_compression_format;
  auto const secondary_format                 = _compression_config.secondary_compression_format;
  auto const secondary_enabled                = secondary_format != gqe::compression_format::none;
  const bool data_is_compressed = _data_buffer.compression_format != gqe::compression_format::none;
  fixed_width_stats.num_primary_compressed_row_groups =
    (data_is_compressed && _data_buffer.compression_format == primary_format) ? 1 : 0;
  fixed_width_stats.num_secondary_compressed_row_groups =
    (data_is_compressed && secondary_enabled && _data_buffer.compression_format == secondary_format)
      ? 1
      : 0;

  fixed_width_stats.num_compressed_row_groups = data_is_compressed ? 1 : 0;
  return column_compression_statistics(fixed_width_stats);
}

template <bool large_string_mode, class Allocator>
string_compressed_sliced_column_storage<large_string_mode, Allocator>::
  string_compressed_sliced_column_storage(cudf::column&& cudf_column,
                                          int partition_size,
                                          memory_kind::type memory_kind,
                                          gqe::compression::configuration compression_config,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr,
                                          allocator_type const& alloc)
  : compressed_sliced_column_storage<Allocator>(
      cudf_column, partition_size, std::move(compression_config), alloc),
    _offset_buffer(offset_element_type, alloc),
    _partition_char_array_sizes(gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc))
{
  compress(std::move(cudf_column), memory_kind, stream, mr);
}

template <bool large_string_mode, class Allocator>
bool string_compressed_sliced_column_storage<large_string_mode, Allocator>::is_large_string() const
{
  return large_string_mode;
}

/*
 * This is similar to the compress method for the regular compressed sliced column
 * but we need to compress the offsets separately and the char arrays for each slice are variably
 * sized based on the total string size for each slice
 *
 * We define each slice as a contiguous chunk of rows of partition_size
 * There's then an associated character array sized as offset((ix_partition+1) * partition_size) -
 * offset(ix_partition * partition_size) An offset array with one offset per row, where the offset
 * indicates the start position of the row in the string This is different from arrow, where there
 * is one more offset than row with the last offset holding the total string size
 *
 * When decompressing, we'll combine the strings from all the unpruned slices and adjust the offsets
 * to point correctly into the combined char array
 *
 * We'll also make this into valid arrow by adding an extra offset that indicates the total string
 * size
 */
template <bool large_string_mode, class Allocator>
void string_compressed_sliced_column_storage<large_string_mode, Allocator>::compress(
  cudf::column&& cudf_column,
  memory_kind::type memory_kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  const size_t num_rows        = cudf_column.size();
  const size_t num_partitions  = gqe::utility::divide_round_up(num_rows, this->_partition_size);
  const auto column_null_count = cudf_column.null_count();

  // First we'll compress the character array
  // These are variable length depending on the row lengths in each partition
  //
  auto column_content = cudf_column.release();
  auto& child_column  = column_content.children.front();
  auto child_content  = child_column->release();

  uint8_t* char_data         = reinterpret_cast<uint8_t*>(column_content.data->data());
  offsets_type* offsets_data = reinterpret_cast<offsets_type*>(child_content.data->data());
  std::vector<offsets_type> offsets_data_host(num_rows + 1);
  GQE_CUDA_TRY(cudaMemcpy(
    offsets_data_host.data(), offsets_data, child_content.data->size(), cudaMemcpyDefault));

  std::vector<rmm::device_buffer> device_uncompressed_data_buffers;
  std::vector<const void*> input_ptrs;
  std::vector<size_t> input_sizes;
  std::vector<void*> device_ptrs;

  device_uncompressed_data_buffers.reserve(num_partitions);
  input_ptrs.reserve(num_partitions);
  input_sizes.reserve(num_partitions);
  device_ptrs.reserve(num_partitions);
  _partition_char_array_sizes.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t start_offset        = ix_partition * this->_partition_size;
    size_t end_offset          = std::min(start_offset + this->_partition_size, num_rows);
    size_t partition_char_size = offsets_data_host[end_offset] - offsets_data_host[start_offset];
    device_uncompressed_data_buffers.emplace_back(partition_char_size, stream, mr);
    input_ptrs.push_back(
      reinterpret_cast<const void*>(char_data + offsets_data_host[start_offset]));
    input_sizes.push_back(partition_char_size);
    device_ptrs.push_back(reinterpret_cast<void*>(device_uncompressed_data_buffers.back().data()));
    _partition_char_array_sizes.push_back(partition_char_size);
  }
  gqe::utility::do_batched_memcpy((void**)device_ptrs.data(),
                                  (void**)input_ptrs.data(),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream);

  std::vector<cudf::size_type> chars_compressed_sizes;
  auto const chars_data_type =
    gqe::compression::get_optimal_nvcomp_data_type(this->_data_buffer.element_type.id());
  auto chars_buffers =
    gqe::compression::compress_batch(this->_compression_config,
                                     chars_data_type,
                                     std::move(device_uncompressed_data_buffers),
                                     this->_data_buffer.compression_format,
                                     this->_data_buffer.compressed_size,
                                     this->_data_buffer.uncompressed_size,
                                     this->_data_buffer.primary_compressed_size,
                                     this->_data_buffer.secondary_compressed_size,
                                     chars_compressed_sizes,
                                     memory_kind,
                                     true /*try secondary compression*/,
                                     stream,
                                     mr);
  // rmm::device_buffer does not expose ownership transfer for its allocation, so persist each
  // compressed partition with an explicit copy into gqe::storage::buffer storage.
  this->_data_buffer.buffers.reserve(chars_buffers.size());
  for (auto const& chars_buffer : chars_buffers) {
    this->_data_buffer.buffers.emplace_back(chars_buffer, stream, mr);
  }

  // Compress the null mask using the parent class method
  if (column_null_count > 0) {
    // Use do_compress here because the null mask is the same as in the normal compressed sliced
    // column Char buffers and offset arrays are set up differently for strings and can't use the
    // shared helper.
    this->do_compress(column_content.null_mask.get(),
                      this->_null_mask_buffer,
                      num_rows,
                      num_partitions,
                      true /*is_null_mask*/,
                      memory_kind,
                      stream,
                      mr);
    rmm::device_buffer null_mask_buffer(
      column_content.null_mask->data(), column_content.null_mask->size(), stream, mr);
    this->_null_counts.reserve(num_partitions);
    for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
      // Get the null count for this partition
      size_t start_ix = ix_partition * this->_partition_size;
      size_t stop_ix  = std::min(start_ix + this->_partition_size, num_rows);
      this->_null_counts.push_back(cudf::null_count(
        reinterpret_cast<const uint32_t*>(null_mask_buffer.data()), start_ix, stop_ix, stream));
    }
  }

  // Adjust the offset array and compress it. We can do the offset adjustment on the CPU because
  // the write task isn't timed.
  input_ptrs.clear();
  input_sizes.clear();
  device_ptrs.clear();
  std::vector<rmm::device_buffer> device_uncompressed_offsets_buffers;
  device_uncompressed_offsets_buffers.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ++ix_partition) {
    size_t ix_start       = ix_partition * this->_partition_size;
    size_t ix_end         = std::min(ix_start + this->_partition_size, num_rows);
    size_t partition_rows = ix_end - ix_start;
    size_t partition_base = offsets_data_host[ix_start];
    // We'll adjust each offset such that for the partition, the offsets start at zero
    for (size_t ix_offset = ix_start; ix_offset < ix_end; ++ix_offset) {
      offsets_data_host[ix_offset] -= partition_base;
    }
    input_ptrs.push_back(reinterpret_cast<const void*>(&offsets_data_host[ix_start]));
    input_sizes.push_back(partition_rows * sizeof(offsets_type));
    device_uncompressed_offsets_buffers.emplace_back(
      partition_rows * sizeof(offsets_type), stream, mr);
    device_ptrs.push_back(
      reinterpret_cast<void*>(device_uncompressed_offsets_buffers.back().data()));
  }
  gqe::utility::do_batched_memcpy((void**)device_ptrs.data(),
                                  (void**)input_ptrs.data(),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream);

  std::vector<cudf::size_type> offsets_compressed_sizes;
  auto const offsets_data_type =
    gqe::compression::get_optimal_nvcomp_data_type(_offset_buffer.element_type.id());
  auto offsets_buffers =
    gqe::compression::compress_batch(this->_compression_config,
                                     offsets_data_type,
                                     std::move(device_uncompressed_offsets_buffers),
                                     _offset_buffer.compression_format,
                                     _offset_buffer.compressed_size,
                                     _offset_buffer.uncompressed_size,
                                     _offset_buffer.primary_compressed_size,
                                     _offset_buffer.secondary_compressed_size,
                                     offsets_compressed_sizes,
                                     memory_kind,
                                     true /* try secondary compression */,
                                     stream,
                                     mr);
  // rmm::device_buffer does not expose ownership transfer for its allocation, so persist each
  // compressed partition with an explicit copy into gqe::storage::buffer storage.
  _offset_buffer.buffers.reserve(offsets_buffers.size());
  for (auto const& offsets_buffer : offsets_buffers) {
    _offset_buffer.buffers.emplace_back(offsets_buffer, stream, mr);
  }

  stream.synchronize();
}

template <bool large_string_mode, class Allocator>
size_t string_compressed_sliced_column_storage<large_string_mode, Allocator>::
  get_char_partition_output_size(size_t partition_idx) const
{
  return _partition_char_array_sizes[partition_idx];
}

template <bool large_string_mode, class Allocator>
int64_t string_compressed_sliced_column_storage<large_string_mode, Allocator>::get_compressed_size()
  const
{
  return static_cast<int64_t>(this->_data_buffer.compressed_size + _offset_buffer.compressed_size +
                              this->_null_mask_buffer.compressed_size);
}

template <bool large_string_mode, class Allocator>
int64_t
string_compressed_sliced_column_storage<large_string_mode, Allocator>::get_uncompressed_size() const
{
  return static_cast<int64_t>(this->_data_buffer.uncompressed_size +
                              _offset_buffer.uncompressed_size +
                              this->_null_mask_buffer.uncompressed_size);
}

template <bool large_string_mode, class Allocator>
bool string_compressed_sliced_column_storage<large_string_mode, Allocator>::is_compressed() const
{
  return this->_data_buffer.compression_format != gqe::compression_format::none ||
         _offset_buffer.compression_format != gqe::compression_format::none;
}

template <bool large_string_mode, class Allocator>
column_compression_statistics
string_compressed_sliced_column_storage<large_string_mode, Allocator>::get_compression_stats() const
{
  string_compression_statistics string_stats;
  auto const chars_compressed_size     = this->_data_buffer.compressed_size;
  auto const chars_uncompressed_size   = this->_data_buffer.uncompressed_size;
  auto const offsets_compressed_size   = _offset_buffer.compressed_size;
  auto const offsets_uncompressed_size = _offset_buffer.uncompressed_size;

  // Offsets buffer statistics
  string_stats.offsets_stats.compressed_size   = static_cast<int64_t>(offsets_compressed_size);
  string_stats.offsets_stats.uncompressed_size = static_cast<int64_t>(offsets_uncompressed_size);
  string_stats.offsets_stats.primary_compressed_size =
    static_cast<int64_t>(_offset_buffer.primary_compressed_size);
  string_stats.offsets_stats.secondary_compressed_size =
    static_cast<int64_t>(_offset_buffer.secondary_compressed_size);
  auto const primary_format    = this->_compression_config.primary_compression_format;
  auto const secondary_format  = this->_compression_config.secondary_compression_format;
  auto const secondary_enabled = secondary_format != gqe::compression_format::none;
  const bool offsets_are_compressed =
    _offset_buffer.compression_format != gqe::compression_format::none;
  string_stats.offsets_stats.num_compressed_row_groups = offsets_are_compressed ? 1ul : 0;
  string_stats.offsets_stats.num_primary_compressed_row_groups =
    (offsets_are_compressed && _offset_buffer.compression_format == primary_format) ? 1ul : 0;
  string_stats.offsets_stats.num_secondary_compressed_row_groups =
    (offsets_are_compressed && secondary_enabled &&
     _offset_buffer.compression_format == secondary_format)
      ? 1ul
      : 0;

  // Chars buffer statistics
  string_stats.chars_stats.compressed_size   = static_cast<int64_t>(chars_compressed_size);
  string_stats.chars_stats.uncompressed_size = static_cast<int64_t>(chars_uncompressed_size);
  string_stats.chars_stats.primary_compressed_size =
    static_cast<int64_t>(this->_data_buffer.primary_compressed_size);
  string_stats.chars_stats.secondary_compressed_size =
    static_cast<int64_t>(this->_data_buffer.secondary_compressed_size);
  const bool chars_are_compressed =
    this->_data_buffer.compression_format != gqe::compression_format::none;
  string_stats.chars_stats.num_compressed_row_groups = chars_are_compressed ? 1ul : 0;
  string_stats.chars_stats.num_primary_compressed_row_groups =
    (chars_are_compressed && this->_data_buffer.compression_format == primary_format) ? 1ul : 0;
  string_stats.chars_stats.num_secondary_compressed_row_groups =
    (chars_are_compressed && secondary_enabled &&
     this->_data_buffer.compression_format == secondary_format)
      ? 1ul
      : 0;

  return column_compression_statistics(string_stats);
}

template <bool large_string_mode, class Allocator>
void string_compressed_sliced_column_storage<large_string_mode, Allocator>::deallocate(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // TODO: Fix system_memory_resource, pinned_memory_resource, numa_memory_resource, and
  // boost_shared_memory_resource, which immediately free host/shared memory while ignoring the
  // stream. Remove this once they satisfy RMM stream-ordered semantics.
  stream.synchronize();
  this->_data_buffer.deallocate(stream, mr);
  this->_null_mask_buffer.deallocate(stream, mr);
  _offset_buffer.deallocate(stream, mr);
}

void compressed_sliced_column_interface::serialize_to_disk(std::string const& column_name,
                                                           std::filesystem::path const& file_path,
                                                           size_t row_group_index,
                                                           rmm::cuda_stream_view stream) const
{
  (void)column_name;
  (void)file_path;
  (void)row_group_index;
  (void)stream;
  throw std::runtime_error("Serialize not implemented");
}

std::unique_ptr<compressed_sliced_column> compressed_sliced_column::deserialize_from_disk(
  std::filesystem::path const& file_path,
  std::string const& column_name,
  size_t row_group_index,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto storage = compressed_sliced_column::storage_type::deserialize_from_disk(
    file_path, column_name, row_group_index, stream, mr, allocator_type{});
  return std::unique_ptr<compressed_sliced_column>(
    new compressed_sliced_column(std::move(storage)));
}

template class compressed_sliced_column_storage<std::allocator<std::byte>>;
template class compressed_sliced_column_storage<shared_memory_allocator>;
template class string_compressed_sliced_column_storage<false, std::allocator<std::byte>>;
template class string_compressed_sliced_column_storage<true, std::allocator<std::byte>>;
template class string_compressed_sliced_column_storage<false, shared_memory_allocator>;
template class string_compressed_sliced_column_storage<true, shared_memory_allocator>;

}  // namespace storage
}  // namespace gqe
