/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/storage/in_memory.hpp>

#include "in_memory_common.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/expression/json_formatter.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/oneshot_in_memory_reader.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/multi_process_helpers.hpp>
#include <gqe/utility/serialization.hpp>

#include <cudf_test/default_stream.hpp>

#include <cuda/__barrier/barrier_arrive_tx.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <fmt/format.h>
#include <nvcomp.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // std::unique_lock, std::scoped_lock
#include <ranges>
#include <shared_mutex>  // std::shared_lock
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

namespace storage {

// Get the size of a string column in bytes, can only be called on string columns. Returns a pair of
// (chars_size, offsets_size).
template <typename ColumnStorage>
std::pair<int64_t, int64_t> get_string_column_size(ColumnStorage const& column)
{
  if (column.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("get_string_column_size can only be called on string columns");
  }
  if (column.num_children() < 1) {
    throw std::runtime_error("String column storage must contain an offsets child");
  }
  auto const& offsets  = column.child(0);
  int64_t chars_size   = static_cast<int64_t>(column.data_size());
  int64_t offsets_size = static_cast<int64_t>(offsets.data_size());
  return std::make_pair(chars_size, offsets_size);
}

// Get the size of a column in bytes, string column includes both chars and offsets size.
template <typename ColumnStorage>
int64_t get_column_size(ColumnStorage const& column)
{
  auto const type = column.type();
  if (type.id() == cudf::type_id::STRING) {
    auto [chars_size, offsets_size] = get_string_column_size(column);
    return chars_size + offsets_size;
  }
  return static_cast<int64_t>(column.data_size());
}

// Support logging of column_layout
constexpr auto format_as(const column_layout layout)
{
  switch (layout) {
    case column_layout::contiguous: return "contiguous";
    case column_layout::compressed_sliced: return "compressed_sliced";
    case column_layout::string_compressed_sliced: return "string_compressed_sliced";
    case column_layout::large_string_compressed_sliced: return "large_string_compressed_sliced";
  }
  // Throw an exception so this will get fixed quickly when we add a new type.
  throw std::runtime_error("column_layout not supported for log formatting");
}

// Cast a column from a row group to a desired type
template <typename T>
const T& get_column(const row_group_interface* row_group, size_t column_idx)
{
  const auto& column_base = row_group->get_column(column_idx);
  const T& column         = static_cast<const T&>(column_base);
  return column;
}

// Common implementation for local and shared contiguous column wrappers.

template <class StorageHandle>
int64_t basic_contiguous_column<StorageHandle>::size() const
{
  assert(static_cast<std::size_t>(_storage_handle.get().size()) <=
         std::numeric_limits<int64_t>::max());
  return static_cast<int64_t>(_storage_handle.get().size());
}

template <class StorageHandle>
cudf::size_type basic_contiguous_column<StorageHandle>::null_count() const
{
  return _storage_handle.get().null_count();
}

template <class StorageHandle>
bool basic_contiguous_column<StorageHandle>::is_compressed() const
{
  return false;
}

template <class StorageHandle>
int64_t basic_contiguous_column<StorageHandle>::get_compressed_size() const
{
  return get_uncompressed_size();
}

template <class StorageHandle>
int64_t basic_contiguous_column<StorageHandle>::get_uncompressed_size() const
{
  return get_column_size(_storage_handle.get());
}

template <class StorageHandle>
column_compression_statistics basic_contiguous_column<StorageHandle>::get_compression_stats() const
{
  auto const& column = _storage_handle.get();
  if (column.type().id() == cudf::type_id::STRING) {
    string_compression_statistics string_stats;
    auto [chars_size, offsets_size]              = get_string_column_size(column);
    string_stats.offsets_stats.compressed_size   = offsets_size;
    string_stats.offsets_stats.uncompressed_size = offsets_size;
    string_stats.chars_stats.compressed_size     = chars_size;
    string_stats.chars_stats.uncompressed_size   = chars_size;

    return column_compression_statistics(string_stats);
  }

  fixed_width_compression_statistics fixed_width_stats;
  fixed_width_stats.compressed_size           = get_column_size(column);
  fixed_width_stats.uncompressed_size         = fixed_width_stats.compressed_size;
  fixed_width_stats.primary_compressed_size   = fixed_width_stats.compressed_size;
  fixed_width_stats.secondary_compressed_size = fixed_width_stats.uncompressed_size;

  return column_compression_statistics(fixed_width_stats);
}

template <class StorageHandle>
cudf::column_view basic_contiguous_column<StorageHandle>::view() const
{
  return _storage_handle.get().view();
}

template <class StorageHandle>
void basic_contiguous_column<StorageHandle>::deallocate(rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  _storage_handle.deallocate(stream, mr);
}

// Common implementation for local and shared compressed-sliced column wrappers.

template <class StorageHandle>
int64_t basic_compressed_sliced_column<StorageHandle>::size() const
{
  return _storage_handle.get().size();
}

template <class StorageHandle>
cudf::size_type basic_compressed_sliced_column<StorageHandle>::null_count() const
{
  return _storage_handle.get().null_count();
}

template <class StorageHandle>
bool basic_compressed_sliced_column<StorageHandle>::is_compressed() const
{
  return _storage_handle.get().is_compressed();
}

template <class StorageHandle>
int64_t basic_compressed_sliced_column<StorageHandle>::get_compressed_size() const
{
  return _storage_handle.get().get_compressed_size();
}

template <class StorageHandle>
int64_t basic_compressed_sliced_column<StorageHandle>::get_uncompressed_size() const
{
  return _storage_handle.get().get_uncompressed_size();
}

template <class StorageHandle>
column_compression_statistics basic_compressed_sliced_column<StorageHandle>::get_compression_stats()
  const
{
  return _storage_handle.get().get_compression_stats();
}

template <class StorageHandle>
typename basic_compressed_sliced_column<StorageHandle>::buffer_view
basic_compressed_sliced_column<StorageHandle>::data_buffer_view() const
{
  return _storage_handle.get().data_buffer_view();
}

template <class StorageHandle>
gqe::compression::configuration const&
basic_compressed_sliced_column<StorageHandle>::get_compression_config() const
{
  return _storage_handle.get().get_compression_config();
}

template <class StorageHandle>
void basic_compressed_sliced_column<StorageHandle>::deallocate(rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  _storage_handle.deallocate(stream, mr);
}

// Common implementation for local and shared string compressed-sliced column wrappers.

template <bool large_string_mode, class StorageHandle>
column_layout basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::layout()
  const
{
  if constexpr (large_string_mode) {
    return column_layout::large_string_compressed_sliced;
  } else {
    return column_layout::string_compressed_sliced;
  }
}

template <bool large_string_mode, class StorageHandle>
int64_t basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::size() const
{
  return _storage_handle.get().size();
}

template <bool large_string_mode, class StorageHandle>
cudf::size_type
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::null_count() const
{
  return _storage_handle.get().null_count();
}

template <bool large_string_mode, class StorageHandle>
bool basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::is_compressed() const
{
  return _storage_handle.get().is_compressed();
}

template <bool large_string_mode, class StorageHandle>
int64_t
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::get_compressed_size() const
{
  return _storage_handle.get().get_compressed_size();
}

template <bool large_string_mode, class StorageHandle>
int64_t basic_string_compressed_sliced_column<large_string_mode,
                                              StorageHandle>::get_uncompressed_size() const
{
  return _storage_handle.get().get_uncompressed_size();
}

template <bool large_string_mode, class StorageHandle>
column_compression_statistics
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::get_compression_stats()
  const
{
  return _storage_handle.get().get_compression_stats();
}

template <bool large_string_mode, class StorageHandle>
bool basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::is_large_string()
  const
{
  return _storage_handle.get().is_large_string();
}

template <bool large_string_mode, class StorageHandle>
typename basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::buffer_view
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::data_buffer_view() const
{
  return _storage_handle.get().data_buffer_view();
}

template <bool large_string_mode, class StorageHandle>
typename basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::buffer_view
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::char_buffer_view() const
{
  return data_buffer_view();
}

template <bool large_string_mode, class StorageHandle>
typename basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::buffer_view
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::offset_buffer_view() const
{
  return _storage_handle.get().offset_buffer_view();
}

template <bool large_string_mode, class StorageHandle>
size_t basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::
  get_char_partition_output_size(size_t partition_idx) const
{
  return _storage_handle.get().get_char_partition_output_size(partition_idx);
}

template <bool large_string_mode, class StorageHandle>
gqe::compression::configuration const&
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::get_compression_config()
  const
{
  return _storage_handle.get().get_compression_config();
}

template <bool large_string_mode, class StorageHandle>
void basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::deallocate(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  _storage_handle.deallocate(stream, mr);
}

// Explicit instantiations for the storage handles and wrapper implementations

template class local_storage_handle<contiguous_column_storage>;
template class shared_storage_handle<shared_contiguous_column_storage>;
template class local_storage_handle<compressed_sliced_column_storage<>>;
template class shared_storage_handle<shared_compressed_sliced_column_storage>;
template class local_storage_handle<string_compressed_sliced_column_storage<false>>;
template class local_storage_handle<string_compressed_sliced_column_storage<true>>;
template class shared_storage_handle<shared_string_compressed_sliced_column_storage<false>>;
template class shared_storage_handle<shared_string_compressed_sliced_column_storage<true>>;

template class basic_contiguous_column<local_storage_handle<contiguous_column_storage>>;
template class basic_contiguous_column<shared_storage_handle<shared_contiguous_column_storage>>;
template class basic_compressed_sliced_column<
  local_storage_handle<compressed_sliced_column_storage<>>>;
template class basic_compressed_sliced_column<
  shared_storage_handle<shared_compressed_sliced_column_storage>>;
template class basic_string_compressed_sliced_column<
  false,
  local_storage_handle<string_compressed_sliced_column_storage<false>>>;
template class basic_string_compressed_sliced_column<
  true,
  local_storage_handle<string_compressed_sliced_column_storage<true>>>;
template class basic_string_compressed_sliced_column<
  false,
  shared_storage_handle<shared_string_compressed_sliced_column_storage<false>>>;
template class basic_string_compressed_sliced_column<
  true,
  shared_storage_handle<shared_string_compressed_sliced_column_storage<true>>>;

pruning_result_t::pruning_result_t(const std::vector<zone_map::partition>& partitions,
                                   cudf::size_type partition_size)
  : _partitions(partitions), _partition_size(partition_size)
{
  // Compute candidate partitions
  _candidate_partitions.reserve(partitions.size());
  std::copy_if(partitions.begin(),
               partitions.end(),
               std::back_inserter(_candidate_partitions),
               [](const auto& partition) { return !partition.pruned; });

  // Compute consolidated partitions
  _consolidated_partitions = zone_map::consolidate_partitions(partitions);
  auto it                  = std::remove_if(_consolidated_partitions.begin(),
                           _consolidated_partitions.end(),
                           [](const zone_map::partition& partition) { return partition.pruned; });
  _consolidated_partitions.erase(it, _consolidated_partitions.end());

  // Compute vector of partition indexes when partitioning is enabled. Unpartitioned results keep
  // only row ranges; compressed-sliced materialization requires a positive partition size.
  if (_partition_size > 0) {
    size_t num_partition_indexes = 0;
    for (const auto& partition : _consolidated_partitions) {
      const size_t start_idx = partition.start / _partition_size;
      const size_t end_idx   = gqe::utility::divide_round_up(partition.end, _partition_size);
      num_partition_indexes += end_idx - start_idx;
    }
    _partition_indexes.reserve(num_partition_indexes);

    for (const auto& partition : _consolidated_partitions) {
      size_t start_idx = partition.start / _partition_size;
      size_t end_idx   = gqe::utility::divide_round_up(partition.end, _partition_size);
      for (size_t partition_idx = start_idx; partition_idx < end_idx; ++partition_idx) {
        _partition_indexes.emplace_back(partition_idx);
      }
    }
  }

  // Compute total number of rows
  _num_rows = 0;
  for (const auto& partition : _consolidated_partitions) {
    _num_rows += partition.end - partition.start;
  }
}

cudf::size_type pruning_result_t::partition_size() const { return _partition_size; }

const std::vector<zone_map::partition>& pruning_result_t::candidate_partitions() const
{
  return _candidate_partitions;
}

const std::vector<zone_map::partition>& pruning_result_t::consolidated_partitions() const
{
  return _consolidated_partitions;
}

const std::vector<size_t>& pruning_result_t::partition_indexes() const
{
  return _partition_indexes;
}

cudf::size_type pruning_result_t::num_rows() const { return _num_rows; }

column_storage_handle::column_storage_handle(column_layout layout_,
                                             shared_memory_handle_t storage_handle_)
  : layout(layout_), storage_handle(storage_handle_)
{
}

shared_row_group_storage::shared_row_group_storage(cudf::size_type num_rows_,
                                                   allocator_type const& alloc)
  : num_rows(num_rows_), columns(rebind_allocator<column_storage_handle>(alloc))
{
}

shared_in_memory_table_storage::shared_in_memory_table_storage(allocator_type const& alloc)
  : _row_group_storage_handles(rebind_allocator<shared_memory_handle_t>(alloc))
{
}

shared_memory_handle_t shared_in_memory_table_storage::append(
  boost::interprocess::managed_shared_memory& segment, shared_row_group_storage&& row_group)
{
  boost::interprocess::scoped_lock lock(_mutex);
  auto* storage_ptr = segment.template construct<shared_row_group_storage>(
    boost::interprocess::anonymous_instance)(std::move(row_group));
  try {
    auto const handle = segment.get_handle_from_address(storage_ptr);
    _row_group_storage_handles.push_back(handle);
    ++_storage_revision;
    return handle;
  } catch (...) {
    segment.destroy_ptr(storage_ptr);
    throw;
  }
}

shared_in_memory_table::row_group_appender::row_group_appender(
  boost::interprocess::managed_shared_memory* non_owning_segment,
  shared_memory_handle_t table_handle)
  : _non_owning_segment(non_owning_segment), _table_handle(table_handle)
{
}

void shared_in_memory_table::row_group_appender::operator()(shared_row_group_storage&& row_group)
{
  if (_non_owning_segment == nullptr || _table_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared row-group appender is not initialized");
  }
  auto* table_storage = static_cast<shared_in_memory_table_storage*>(
    _non_owning_segment->get_address_from_handle(_table_handle));
  table_storage->append(*_non_owning_segment, std::move(row_group));
}

boost::interprocess::managed_shared_memory& shared_in_memory_table::row_group_appender::segment()
  const
{
  if (_non_owning_segment == nullptr) {
    throw std::logic_error("Shared row-group appender segment is not initialized");
  }
  return *_non_owning_segment;
}

contiguous_column::contiguous_column(cudf::column&& cudf_column)
  : contiguous_column(std::move(cudf_column),
                      cudf::get_default_stream(),
                      rmm::mr::get_current_device_resource_ref())
{
}

contiguous_column::contiguous_column(cudf::column&& cudf_column,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr,
                                     storage_type::allocator_type const& alloc)
  : base_type(cudf_column.view(), stream, mr, alloc)
{
}

cudf::mutable_column_view contiguous_column::mutable_view()
{
  return _storage_handle.get().mutable_view();
}

row_group::row_group(cudf::size_type num_rows,
                     std::vector<std::unique_ptr<column_base>>&& columns,
                     std::unique_ptr<gqe::zone_map> zone_map)
  : _num_rows(num_rows), _columns(std::move(columns)), _zone_map(std::move(zone_map))
{
}

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns,
                     std::unique_ptr<gqe::zone_map> zone_map)
  : row_group(columns.empty() ? 0 : static_cast<cudf::size_type>(columns.front()->size()),
              std::move(columns),
              std::move(zone_map))
{
}

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns)
  : row_group(std::move(columns), nullptr)
{
}

int64_t row_group::size() const { return _num_rows; }

int64_t row_group::num_columns() const { return _columns.size(); }

column_base& row_group::get_column(cudf::size_type column_index) const
{
  return *_columns.at(column_index);
}

gqe::zone_map* row_group::zone_map() const { return _zone_map.get(); }

void row_group::deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  for (auto& column : _columns) {
    column->deallocate(stream, mr);
  }
  _columns.clear();
  if (_zone_map) { _zone_map->deallocate(stream, mr); }
  _zone_map.reset();
  _num_rows = 0;
}

shared_row_group::shared_row_group(boost::interprocess::managed_shared_memory& segment,
                                   shared_memory_handle_t shared_row_group_storage_handle)
  : _segment(&segment), _shared_row_group_storage_handle(shared_row_group_storage_handle)
{
  auto const& shared_storage = storage();
  _columns.reserve(shared_storage.columns.size());
  for (auto const& column : shared_storage.columns) {
    switch (column.layout) {
      case column_layout::contiguous:
        _columns.push_back(
          std::make_unique<shared_contiguous_column>(*_segment, column.storage_handle));
        break;
      case column_layout::compressed_sliced:
        _columns.push_back(
          std::make_unique<shared_compressed_sliced_column>(*_segment, column.storage_handle));
        break;
      case column_layout::string_compressed_sliced:
        _columns.push_back(std::make_unique<shared_string_compressed_sliced_column<false>>(
          *_segment, column.storage_handle));
        break;
      case column_layout::large_string_compressed_sliced:
        _columns.push_back(std::make_unique<shared_string_compressed_sliced_column<true>>(
          *_segment, column.storage_handle));
        break;
      default: throw std::logic_error("Unsupported shared column storage layout");
    }
  }
  if (shared_storage.zone_map) {
    _zone_map = std::make_unique<gqe::shared_zone_map>(*_segment, *shared_storage.zone_map);
  }
}

shared_row_group_storage const& shared_row_group::storage() const
{
  if (_segment == nullptr || _shared_row_group_storage_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared row group storage is not initialized");
  }
  return *static_cast<shared_row_group_storage const*>(
    _segment->get_address_from_handle(_shared_row_group_storage_handle));
}

int64_t shared_row_group::size() const { return storage().num_rows; }

int64_t shared_row_group::num_columns() const { return _columns.size(); }

column_base& shared_row_group::get_column(cudf::size_type column_index) const
{
  return *_columns.at(column_index);
}

gqe::zone_map* shared_row_group::zone_map() const { return _zone_map.get(); }

void shared_row_group::deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (_segment == nullptr || _shared_row_group_storage_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared row group storage is not initialized");
  }

  auto* row_group_storage = static_cast<shared_row_group_storage*>(
    _segment->get_address_from_handle(_shared_row_group_storage_handle));

  for (auto& column : _columns) {
    column->deallocate(stream, mr);
  }
  _columns.clear();
  if (_zone_map) { _zone_map->deallocate(stream, mr); }
  _zone_map.reset();

  _segment->destroy_ptr(row_group_storage);
  _segment                         = nullptr;
  _shared_row_group_storage_handle = shared_memory_handle_t{};
}

in_memory_table::in_memory_table(memory_kind::type memory_kind,
                                 std::vector<std::string> const& column_names,
                                 std::vector<cudf::data_type> const& column_types,
                                 task_manager_context* ctx)
  : in_memory_table_interface(),
    _task_manager_context(ctx),
    _memory_kind(memory_kind),
    _column_name_to_index(),
    _column_names(column_names),
    _column_types(column_types),
    _row_groups(),
    _row_group_latch()
{
  if (!_task_manager_context) {
    throw std::invalid_argument("in_memory_table requires a non-null task_manager_context");
  }

  // Populate column name-to-index map
  _column_name_to_index.reserve(column_names.size());
  for (decltype(column_names.size()) idx = 0; idx < column_names.size(); ++idx) {
    _column_name_to_index[column_names[idx]] = idx;
  }
}

in_memory_table::~in_memory_table() = default;

void in_memory_table::deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  std::unique_lock latch_guard(_row_group_latch);
  for (auto& row_group : _row_groups) {
    row_group.deallocate(stream, mr);
  }
  _row_groups.clear();
}

shared_in_memory_table::shared_in_memory_table(
  boost::interprocess::managed_shared_memory* non_owning_segment,
  std::vector<std::string> const& column_names,
  std::vector<cudf::data_type> const& column_types,
  shared_memory_handle_t table_handle)
  : in_memory_table_interface(),
    _memory_kind(memory_kind::boost_shared{}),
    _column_types(column_types),
    _row_groups(),
    _row_group_latch(),
    _shared_segment(non_owning_segment),
    _shared_table_handle(table_handle)
{
  _column_name_to_index.reserve(column_names.size());
  for (decltype(column_names.size()) idx = 0; idx < column_names.size(); ++idx) {
    _column_name_to_index[column_names[idx]] = idx;
  }
  if (_shared_segment == nullptr) {
    throw std::invalid_argument("Cannot attach boost shared in-memory table without segment");
  }
  if (_shared_table_handle == shared_memory_handle_t{}) {
    throw std::invalid_argument("Cannot attach boost shared in-memory table without table handle");
  }
  std::vector<shared_memory_handle_t> row_group_storage_handles;
  {
    auto const& table_storage = storage();
    boost::interprocess::scoped_lock lock(table_storage._mutex);
    row_group_storage_handles.assign(table_storage._row_group_storage_handles.begin(),
                                     table_storage._row_group_storage_handles.end());
  }
  for (auto const shared_row_group_storage_handle : row_group_storage_handles) {
    _row_groups.emplace_back(*_shared_segment, shared_row_group_storage_handle);
  }
}

void shared_in_memory_table::deallocate(rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  std::vector<shared_memory_handle_t> row_group_storage_handles;
  auto& storage_ref = storage();
  {
    boost::interprocess::scoped_lock shared_lock(storage_ref._mutex);
    row_group_storage_handles.assign(storage_ref._row_group_storage_handles.begin(),
                                     storage_ref._row_group_storage_handles.end());
  }
  {
    std::unique_lock local_lock(_row_group_latch);
    _row_groups.clear();
  }

  for (auto const handle : row_group_storage_handles) {
    shared_row_group row_group(*_shared_segment, handle);
    row_group.deallocate(stream, mr);
  }

  _shared_segment->destroy_ptr(&storage_ref);
  _shared_table_handle = shared_memory_handle_t{};
}

std::vector<shared_memory_handle_t> shared_in_memory_table::replace_row_groups(
  shared_row_group_storage&& new_row_group)
{
  auto& storage_ref = storage();

  // Construct the replacement row-group storage first so that a failure below leaves the table
  // unchanged (strong exception guarantee).
  auto* new_storage_ptr = _shared_segment->construct<shared_row_group_storage>(
    boost::interprocess::anonymous_instance)(std::move(new_row_group));

  std::vector<shared_memory_handle_t> old_handles;
  try {
    auto const new_handle = _shared_segment->get_handle_from_address(new_storage_ptr);

    boost::interprocess::scoped_lock lock(storage_ref._mutex);
    // Reserve before the destructive clear/push_back so the push_back cannot reallocate (and thus
    // cannot throw) after we have already dropped the old handles.
    storage_ref._row_group_storage_handles.reserve(1);
    old_handles.assign(storage_ref._row_group_storage_handles.begin(),
                       storage_ref._row_group_storage_handles.end());
    storage_ref._row_group_storage_handles.clear();
    storage_ref._row_group_storage_handles.push_back(new_handle);
    ++storage_ref._storage_revision;
  } catch (...) {
    _shared_segment->destroy_ptr(new_storage_ptr);
    throw;
  }

  return old_handles;
}

void shared_in_memory_table::cleanup_detached_row_groups(
  std::vector<shared_memory_handle_t> const& handles,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) noexcept
{
  // Drop stale local wrappers before destroying the shared storage they reference. If locking
  // fails, retain both the wrappers and detached storage rather than leaving dangling wrappers.
  try {
    std::unique_lock local_lock(_row_group_latch);
    _row_groups.clear();
  } catch (std::exception const& error) {
    GQE_LOG_ERROR("Failed to drop local row-group wrappers after publication: {}", error.what());
    return;
  } catch (...) {
    GQE_LOG_ERROR("Failed to drop local row-group wrappers after publication: unknown error");
    return;
  }

  for (auto const handle : handles) {
    try {
      shared_row_group row_group(*_shared_segment, handle);
      row_group.deallocate(stream, mr);
    } catch (std::exception const& error) {
      GQE_LOG_ERROR("Failed to reclaim detached shared row group: {}", error.what());
    } catch (...) {
      GQE_LOG_ERROR("Failed to reclaim detached shared row group: unknown error");
    }
  }
}

void shared_in_memory_table::clear_row_groups(rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto& storage_ref = storage();

  std::vector<shared_memory_handle_t> old_handles;
  {
    boost::interprocess::scoped_lock lock(storage_ref._mutex);
    old_handles.assign(storage_ref._row_group_storage_handles.begin(),
                       storage_ref._row_group_storage_handles.end());
    storage_ref._row_group_storage_handles.clear();
    ++storage_ref._storage_revision;
  }

  cleanup_detached_row_groups(old_handles, stream, mr);
}

void shared_in_memory_table::replace_contents(cudf::table_view input_table,
                                              rmm::device_async_resource_ref memory_resource,
                                              rmm::cuda_stream_view stream)
{
  // A zero-row (or zero-column) input leaves the table empty: a zero-row column cannot be stored
  // as a shared column.
  if (input_table.num_columns() == 0 || input_table.num_rows() == 0) {
    clear_row_groups(stream, memory_resource);
    return;
  }

  if (static_cast<std::size_t>(input_table.num_columns()) != _column_types.size()) {
    throw std::invalid_argument(
      "shared_in_memory_table::replace_contents: input column count does not match the table "
      "schema");
  }

  auto& segment = *_shared_segment;
  shared_memory_allocator alloc(segment.get_segment_manager());
  shared_row_group_storage new_row_group(input_table.num_rows(), alloc);
  new_row_group.columns.reserve(input_table.num_columns());

  // Until replace_row_groups() takes ownership, the shared column storage constructed below lives
  // in the segment referenced only by handle. Track the handles so that any failure before the
  // publish completes (column construction, stream sync, or the publish itself) releases them
  // rather than leaking shared/device memory.
  std::vector<shared_memory_handle_t> staged_handles;
  staged_handles.reserve(input_table.num_columns());

  try {
    // Statistics tables are tiny and uncompressed: every column is stored contiguously and no zone
    // map is built (they are never predicate-pruned). Rank 0 constructs the backing storage in the
    // segment; the copies are asynchronous on `stream` and synchronized before publishing.
    for (cudf::size_type column_idx = 0; column_idx < input_table.num_columns(); ++column_idx) {
      auto* column_storage = segment.construct<shared_contiguous_column_storage>(
        boost::interprocess::anonymous_instance)(
        input_table.column(column_idx), stream, memory_resource, alloc);
      auto const handle = segment.get_handle_from_address(column_storage);
      new_row_group.columns.emplace_back(column_layout::contiguous, handle);
      staged_handles.push_back(handle);
    }

    stream.synchronize();
    auto const old_handles = replace_row_groups(std::move(new_row_group));

    // Publication transferred ownership of the staged columns to shared metadata. Release the
    // rollback list before cleanup of the detached old row groups.
    staged_handles.clear();
    cleanup_detached_row_groups(old_handles, stream, memory_resource);
  } catch (...) {
    // replace_row_groups only throws before publication, so these columns remain caller-owned.
    for (auto const handle : staged_handles) {
      shared_contiguous_column staged(segment, handle);
      staged.deallocate(stream, memory_resource);
    }
    throw;
  }
}

shared_in_memory_table_storage const& shared_in_memory_table::storage() const
{
  if (_shared_segment == nullptr || _shared_table_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Boost shared in-memory table storage is not initialized");
  }
  return *static_cast<shared_in_memory_table_storage const*>(
    _shared_segment->get_address_from_handle(_shared_table_handle));
}

shared_in_memory_table_storage& shared_in_memory_table::storage()
{
  if (_shared_segment == nullptr || _shared_table_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Boost shared in-memory table storage is not initialized");
  }
  return *static_cast<shared_in_memory_table_storage*>(
    _shared_segment->get_address_from_handle(_shared_table_handle));
}

namespace {

[[nodiscard]] std::string sanitize_path(std::string value)
{
  for (char& c : value) {
    if (c == '/' || c == '\\') { c = '_'; }
  }
  return value;
}

/** On-disk column snapshot: `<rg_dir>/<sanitized_column_name>.bin`. */
[[nodiscard]] std::filesystem::path serialized_column_path(std::filesystem::path const& rg_dir,
                                                           std::string const& column_name)
{
  return rg_dir / (sanitize_path(column_name) + ".bin");
}

/** Metadata sidecar: `<rg_dir>/<sanitized_column_name>.json`. */
[[nodiscard]] std::filesystem::path serialized_column_metadata_path(
  std::filesystem::path const& rg_dir, std::string const& column_name)
{
  return rg_dir / (sanitize_path(column_name) + ".json");
}

[[nodiscard]] std::string sanitize_json_string(std::string_view value)
{
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '"' || c == '\\') { escaped.push_back('\\'); }
    escaped.push_back(c);
  }
  return escaped;
}

//**
// Write column metadata to a JSON file. Used for serialization.
// @param json_path The path to the JSON file.
// @param column_name The name of the column.
// @param row_group_index The index of the row group.
// @param opt_params The optimization parameters.
// @param column The column to write metadata for.
// @return True if the metadata was written successfully, false otherwise.
// */
void write_column_metadata(std::filesystem::path const& json_path,
                           std::string const& column_name,
                           size_t row_group_index,
                           optimization_parameters const& opt_params,
                           compressed_sliced_column_interface const& column)
{
  auto const format = column.data_buffer_view().compression_format();
  std::ofstream out(json_path);
  if (!out) {
    throw std::runtime_error("Could not open metadata file for writing: " + json_path.string());
  }

  // TODO: Add documentation for version number for Serialization.
  out << fmt::format(
    "{{\n"
    R"(  "serialization_version_major": {},)"
    "\n"
    R"(  "serialization_version_minor": {},)"
    "\n"
    R"(  "column_name": "{}",)"
    "\n"
    R"(  "row_group": {},)"
    "\n"
    R"(  "zone_map_size": {},)"
    "\n"
    R"(  "compression_format": "{}",)"
    "\n"
    R"(  "compression_level": {},)"
    "\n"
    R"(  "uncompressed_size": {},)"
    "\n"
    R"(  "compressed_size": {},)"
    "\n"
    R"(  "is_compressed": {})"
    "\n"
    "}}\n",
    utility::serialization_version_major,
    utility::serialization_version_minor,
    sanitize_json_string(column_name),
    row_group_index,
    opt_params.zone_map_partition_size,
    gqe::to_string(format),
    opt_params.compression_level,
    column.get_uncompressed_size(),
    column.get_compressed_size(),
    column.is_compressed() ? "true" : "false");

  if (!out) { throw std::runtime_error("Failed writing metadata file: " + json_path.string()); }
}

/**
 * @brief Numeric suffix of `rg-<n>` directory names.
 *
 * @throw std::invalid_argument if @p is not `rg-<non-negative-integer>`.
 */
[[nodiscard]] int64_t row_group_directory_index(std::filesystem::path const& p)
{
  const std::string name = p.filename().string();
  static constexpr std::string_view k_prefix{"rg-"};
  if (name.size() <= k_prefix.size() || !name.starts_with(k_prefix)) {
    throw std::invalid_argument("row group directory must be named rg-<index>, got: " + name);
  }

  int64_t n         = 0;
  const auto* first = name.data() + k_prefix.size();
  const auto* last  = name.data() + name.size();
  if (auto const r = std::from_chars(first, last, n); r.ec != std::errc{} || r.ptr != last) {
    throw std::invalid_argument("row group directory index is not a plain integer: " + name);
  }
  if (n < 0) {
    throw std::invalid_argument("row group directory index must be non-negative: " + name);
  }
  return n;
}

/** Column indices to serialize; computed before any disk I/O. */
[[nodiscard]] std::vector<std::vector<size_t>> serializable_column_indices_per_row_group(
  std::deque<row_group> const& row_groups, std::string const& table_name)
{
  std::vector<std::vector<size_t>> indices_per_row_group(row_groups.size());
  for (size_t rg_idx = 0; rg_idx < row_groups.size(); ++rg_idx) {
    auto const ncols = static_cast<size_t>(row_groups[rg_idx].num_columns());
    for (size_t col_idx = 0; col_idx < ncols; ++col_idx) {
      column_base const& col = row_groups[rg_idx].get_column(static_cast<cudf::size_type>(col_idx));
      if (dynamic_cast<compressed_sliced_column_interface const*>(&col) == nullptr) {
        GQE_LOG_TRACE("skipping column idx={} layout={} table='{}'",
                      col_idx,
                      static_cast<int>(col.layout()),
                      table_name);
        continue;
      }
      indices_per_row_group[rg_idx].push_back(col_idx);
    }
  }
  return indices_per_row_group;
}

/** Serialize row-group columns under `table_serialized_data_root/rg-<i>/{compression}/{chunk}/`. */
void serialize_row_group_columns(row_group const& rg,
                                 size_t row_group_index,
                                 std::string const& table_serialized_data_root,
                                 std::string const& table_name,
                                 std::vector<std::string> const& column_names,
                                 std::span<size_t const> column_indices,
                                 optimization_parameters const& opt_params,
                                 rmm::cuda_stream_view stream)
{
  std::filesystem::path const column_chunk_dir = utility::serialized_row_group_column_root(
    std::filesystem::path(table_serialized_data_root), row_group_index, opt_params);
  std::error_code mkdir_ec;
  std::filesystem::create_directories(column_chunk_dir, mkdir_ec);

  GQE_LOG_TRACE(
    "serialize_row_group_columns: begin table='{}' row_group={} table_serialized_data_root='{}' "
    "column_dir='{}' num_columns={} mkdir_ec={}",
    table_name,
    row_group_index,
    table_serialized_data_root,
    column_chunk_dir.string(),
    rg.num_columns(),
    mkdir_ec.value());

  if (mkdir_ec) { throw std::runtime_error("Could not create directories: " + mkdir_ec.message()); }

  for (size_t col_idx : column_indices) {
    auto const& col = dynamic_cast<compressed_sliced_column_interface const&>(
      rg.get_column(static_cast<cudf::size_type>(col_idx)));
    std::string label = column_names[col_idx];

    std::filesystem::path const file_path = serialized_column_path(column_chunk_dir, label);
    std::filesystem::path const metadata_path =
      serialized_column_metadata_path(column_chunk_dir, label);
    GQE_LOG_TRACE("column='{}' idx={} bin='{}' json='{}' table='{}'",
                  label,
                  col_idx,
                  file_path.string(),
                  metadata_path.string(),
                  table_name);
    write_column_metadata(metadata_path, label, row_group_index, opt_params, col);
    col.serialize_to_disk(label, file_path, row_group_index, stream);
  }

  GQE_LOG_TRACE(
    "Serialization completed successfully: table='{}' row_group={}", table_name, row_group_index);
}

/** Serialize row-group zone maps under `table_serialized_data_root/rg-<i>/zone_maps/`. */
void serialize_zone_maps_to_disk(std::deque<row_group> const& row_groups,
                                 std::filesystem::path const& table_serialized_data_root,
                                 std::string const& table_name,
                                 rmm::cuda_stream_view stream)
{
  (void)stream;
  for (size_t i = 0; i < row_groups.size(); ++i) {
    std::filesystem::path const zm_dir =
      utility::serialized_row_group_zone_maps_root(table_serialized_data_root, i);
    std::error_code ec;
    std::filesystem::create_directories(zm_dir, ec);
    if (ec) {
      throw std::runtime_error("Could not create zone_maps directory '" + zm_dir.string() +
                               "': " + ec.message());
    }
    GQE_LOG_TRACE("Created zone_maps directory: {}", zm_dir.string());
    // TODO: write zone map payload for row_groups[i] into zm_dir
  }
  GQE_LOG_TRACE(
    "serialize_zone_maps_to_disk: directories created; payload write not implemented table='{}' "
    "table_serialized_data_root='{}'",
    table_name,
    table_serialized_data_root.string());
  throw std::runtime_error(
    "serialize_zone_maps_to_disk: zone map payload write not implemented for table '" + table_name +
    "'");
}

/**
 * Load row-group zone maps from `table_serialized_data_root/rg-<i>/zone_maps/` before column
 * snapshots (not
 * implemented yet).
 */
void deserialize_zone_maps_from_disk(std::deque<row_group>& row_groups,
                                     std::filesystem::path const& table_serialized_data_root,
                                     std::vector<std::filesystem::path> const& rg_dirs,
                                     std::string const& table_name,
                                     rmm::cuda_stream_view stream)
{
  (void)row_groups;
  (void)table_serialized_data_root;
  (void)rg_dirs;
  (void)stream;
  GQE_LOG_TRACE(
    "deserialize_zone_maps_from_disk: not implemented yet table='{}' "
    "table_serialized_data_root='{}'",
    table_name,
    table_serialized_data_root.string());
  throw std::runtime_error("deserialize_zone_maps_from_disk: not implemented yet for table '" +
                           table_name + "'");
}

}  // namespace

void in_memory_table::deserialize_table_from_disk(std::string const& table_serialized_data_root,
                                                  std::string const& table_name,
                                                  rmm::cuda_stream_view stream)
{
  (void)table_name;
  auto mr = _task_manager_context->get_table_memory_resource(_memory_kind);
  std::unique_lock latch_guard(_row_group_latch);
  for (auto& row_group : _row_groups) {
    row_group.deallocate(stream, mr);
  }
  _row_groups.clear();

  std::filesystem::path const root(table_serialized_data_root);
  if (!std::filesystem::is_directory(root)) {
    throw std::runtime_error("passed path is not a directory: " + table_serialized_data_root);
  }

  std::vector<std::filesystem::path> rg_dirs;
  for (auto const& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_directory()) { continue; }
    std::string const fn = entry.path().filename().string();
    if (fn.starts_with("rg-")) { rg_dirs.push_back(entry.path()); }
  }

  try {
    // `directory_iterator` order is unspecified; load row groups in rg-0, rg-1, … numeric order.
    std::ranges::sort(rg_dirs, {}, row_group_directory_index);

    deserialize_zone_maps_from_disk(_row_groups, root, rg_dirs, table_name, stream);

    auto const& opt_params = _task_manager_context->get_optimization_parameters();

    for (std::filesystem::path const& rg_path : rg_dirs) {
      auto const row_group_index = static_cast<size_t>(row_group_directory_index(rg_path));
      std::filesystem::path const column_chunk_dir =
        utility::serialized_row_group_column_root(root, row_group_index, opt_params);
      std::vector<std::unique_ptr<column_base>> columns;
      columns.reserve(_column_names.size());

      for (std::string const& col_name : _column_names) {
        std::filesystem::path const bin = serialized_column_path(column_chunk_dir, col_name);
        if (!std::filesystem::exists(bin)) {
          throw std::runtime_error("Deserialize: Missing binary file: " + bin.string());
        }
        std::ifstream probe(bin, std::ios::binary);
        std::uint32_t magic = 0;
        probe.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        probe.close();

        // TODO: Update magic number to something standardized.
        if (magic == 0x47514E01u) {
          auto col = compressed_sliced_column::deserialize_from_disk(
            bin, col_name, row_group_index, stream, mr);
          columns.push_back(std::move(col));
        } else {
          throw std::runtime_error("deserialize: unsupported magic " + fmt::format("{:x}", magic) +
                                   " for column '" + col_name + "'");
        }
      }

      _row_groups.push_back(row_group(std::move(columns)));
    }
  } catch (std::invalid_argument const& e) {
    throw std::runtime_error(std::string("Invalid row-group directory under '") + root.string() +
                             "': " + e.what());
  }
}

void in_memory_table::serialize_table_to_disk(std::string const& table_serialized_data_root,
                                              std::string const& table_name,
                                              rmm::cuda_stream_view stream)
{
  std::shared_lock latch_guard(_row_group_latch);
  auto const& opt_params = _task_manager_context->get_optimization_parameters();

  auto const column_indices_per_row_group =
    serializable_column_indices_per_row_group(_row_groups, table_name);

  serialize_zone_maps_to_disk(
    _row_groups, std::filesystem::path(table_serialized_data_root), table_name, stream);

  for (size_t i = 0; i < _row_groups.size(); ++i) {
    serialize_row_group_columns(_row_groups[i],
                                i,
                                table_serialized_data_root,
                                table_name,
                                _column_names,
                                column_indices_per_row_group[i],
                                opt_params,
                                stream);
  }
}

bool in_memory_table::is_readable() const { return true; }

bool in_memory_table::is_writeable() const { return true; }

int32_t in_memory_table::max_concurrent_readers() const
{
  auto const nrow_groups = _row_groups.size();
  assert(nrow_groups <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()));

  return static_cast<int32_t>(nrow_groups);
}

int32_t in_memory_table::max_concurrent_writers() const
{
  // Each writer appends a new row group to the table. Thus, the number of
  // writers is theoretically unlimited.
  return std::numeric_limits<int32_t>::max();
}

std::unique_ptr<readable_view> in_memory_table::readable_view()
{
  return std::unique_ptr<storage::readable_view>(new in_memory_readable_view(this));
}

std::unique_ptr<writeable_view> in_memory_table::writeable_view()
{
  return std::unique_ptr<storage::writeable_view>(new in_memory_writeable_view(this));
}

cudf::size_type in_memory_table::get_column_index(std::string const& column_name) const
{
  return _column_name_to_index.at(column_name);
}

memory_kind::type in_memory_table::get_memory_kind() const { return _memory_kind; }

std::size_t in_memory_table::num_row_groups() const { return _row_groups.size(); }

row_group_interface const* in_memory_table::row_group_at(std::size_t index) const
{
  return &_row_groups.at(index);
}

std::shared_mutex& in_memory_table::row_group_latch() const { return _row_group_latch; }

in_memory_table::row_group_appender in_memory_table::get_row_group_appender()
{
  return {&_row_groups, &_row_group_latch};
}

bool shared_in_memory_table::is_readable() const { return true; }

bool shared_in_memory_table::is_writeable() const { return true; }

std::uint64_t shared_in_memory_table::storage_revision() const
{
  auto const& table_storage = storage();
  boost::interprocess::scoped_lock lock(table_storage._mutex);
  return table_storage._storage_revision;
}

int32_t shared_in_memory_table::max_concurrent_readers() const
{
  std::shared_lock latch_guard(_row_group_latch);
  auto const nrow_groups = _row_groups.size();
  assert(nrow_groups <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()));
  return static_cast<int32_t>(nrow_groups);
}

int32_t shared_in_memory_table::max_concurrent_writers() const
{
  return std::numeric_limits<int32_t>::max();
}

std::unique_ptr<readable_view> shared_in_memory_table::readable_view()
{
  return std::unique_ptr<storage::readable_view>(new in_memory_readable_view(this));
}

std::unique_ptr<writeable_view> shared_in_memory_table::writeable_view()
{
  return std::unique_ptr<storage::writeable_view>(new in_memory_writeable_view(this));
}

shared_in_memory_table::row_group_appender shared_in_memory_table::get_row_group_appender()
{
  return {_shared_segment, _shared_table_handle};
}

shared_memory_handle_t shared_in_memory_table::shared_table_handle() const
{
  return _shared_table_handle;
}

cudf::size_type shared_in_memory_table::get_column_index(std::string const& column_name) const
{
  return _column_name_to_index.at(column_name);
}

memory_kind::type shared_in_memory_table::get_memory_kind() const { return _memory_kind; }

std::size_t shared_in_memory_table::num_row_groups() const { return _row_groups.size(); }

row_group_interface const* shared_in_memory_table::row_group_at(std::size_t index) const
{
  return &_row_groups.at(index);
}

std::shared_mutex& shared_in_memory_table::row_group_latch() const { return _row_group_latch; }

in_memory_table::row_group_appender::row_group_appender(
  std::deque<row_group>* non_owning_row_groups, std::shared_mutex* non_owning_row_group_latch)
  : _non_owning_row_groups(non_owning_row_groups),
    _non_owning_row_group_latch(non_owning_row_group_latch)
{
}

void in_memory_table::row_group_appender::operator()(row_group&& new_row_group)
{
  std::scoped_lock latch_guard(*_non_owning_row_group_latch);
  _non_owning_row_groups->push_back(std::move(new_row_group));
}

void in_memory_table::row_group_appender::operator()(std::vector<row_group>&& new_row_groups)
{
  std::scoped_lock latch_guard(*_non_owning_row_group_latch);
  for (auto&& rg : new_row_groups) {
    _non_owning_row_groups->push_back(std::move(rg));
  }
}

// Helper to create a column view slice
cudf::column_view slice_column(const cudf::column_view& full_column,
                               const cudf::size_type size,
                               const cudf::size_type offset,
                               const cudf::size_type null_counts)
{
  const auto type      = full_column.type();
  const auto head      = full_column.head();
  const auto null_mask = full_column.null_mask();
  const auto children =
    std::vector<cudf::column_view>(full_column.child_begin(), full_column.child_end());
  return cudf::column_view(type, size, head, null_mask, null_counts, offset, children);
}

cudf::size_type detail::checked_row_count(pruning_results_t const& pruning_results)
{
  int64_t total_rows = 0;
  for (auto const& [/* row_group */ _, pruning_result] : pruning_results) {
    auto const rows = static_cast<int64_t>(pruning_result.num_rows());
    GQE_EXPECTS(
      rows >= 0 &&
        total_rows <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()) - rows,
      "in-memory result row count exceeds cudf::size_type",
      std::overflow_error);
    total_rows += rows;
  }
  return static_cast<cudf::size_type>(total_rows);
}

output_column_builder::output_column_builder(cudf::data_type cudf_type,
                                             std::shared_ptr<pruning_results_t> pruning_results,
                                             cudf::size_type column_idx)
  : _cudf_type(cudf_type),
    _pruning_results(std::move(pruning_results)),
    _column_idx(column_idx),
    _num_rows(detail::checked_row_count(*_pruning_results))
{
}

void output_column_builder::submit_materialization_requests(utility::copy_batch&,
                                                            decompression_batch&,
                                                            rmm::cuda_stream_view,
                                                            rmm::device_async_resource_ref)
{
}

std::unique_ptr<cudf::column> concatenating_output_column_builder::make_cudf_column(
  rmm::cuda_stream_view concatenation_stream, rmm::device_async_resource_ref mr)
{
  std::vector<cudf::column_view> pruned_columns;
  pruned_columns.reserve(_pruning_results->size());
  for (auto& [row_group, pruning_result] : *_pruning_results) {
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      auto& column = row_group->get_column(_column_idx);
      if (column.layout() != column_layout::contiguous) {
        throw std::logic_error("Use dedicated compressed_sliced_output_column_builder instead");
      }
      auto const full_column_view = dynamic_cast<contiguous_column_interface&>(column).view();
      // Construct the partitioned column view
      const auto size               = partition.end - partition.start;
      const auto offset             = partition.start;
      const auto null_counts        = partition.null_counts[_column_idx];
      const auto pruned_column_view = slice_column(full_column_view, size, offset, null_counts);
      pruned_columns.push_back(std::move(pruned_column_view));
    }
  }

  // Construct the output column from the (partitioned) row group column views using
  // cudf::concatenate
  std::unique_ptr<cudf::column> cudf_column =
    cudf::concatenate(pruned_columns, concatenation_stream, mr);

  GQE_LOG_DEBUG(
    "Processed column with concatenate: _column_idx = {}, "
    "cudf_column->size() = {}",
    _column_idx,
    cudf_column->size());
  return cudf_column;
}

namespace {

std::unique_ptr<output_column_builder> make_output_column_builder(
  const cudf::data_type cudf_type,
  const column_layout layout,
  std::shared_ptr<pruning_results_t> pruning_results,
  cudf::size_type column_idx,
  const row_group_interface*)
{
  if (layout == column_layout::contiguous && cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<contiguous_output_column_builder<contiguous_column_interface>>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::contiguous &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<concatenating_output_column_builder>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::compressed_sliced &&
             cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<compressed_sliced_output_column_builder>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::string_compressed_sliced &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<string_compressed_sliced_output_column_builder<false>>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::large_string_compressed_sliced &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<string_compressed_sliced_output_column_builder<true>>(
      cudf_type, pruning_results, column_idx);
  }

  return std::make_unique<concatenating_output_column_builder>(
    cudf_type, pruning_results, column_idx);
}

rmm::device_buffer make_output_buffer(size_t buffer_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  return rmm::device_buffer(buffer_size, stream, mr);
}

rmm::device_buffer make_output_buffer(size_t num_values,
                                      size_t value_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  return make_output_buffer(num_values * value_size, stream, mr);
}

void submit_buffer_materialization_requests(
  utility::copy_batch& c_batch,
  decompression_batch& d_batch,
  std::deque<rmm::device_buffer>& staging_buffers,
  compressed_sliced_buffer_view host_source_buffer,
  gqe::compression::configuration const& compression_config,
  std::span<size_t const> partition_indexes,
  std::byte* output_ptr,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (partition_indexes.empty()) { return; }

  if (host_source_buffer.is_compressed()) {
    // Each selected partition is copied into staging, then the whole staging buffer is represented
    // by one decompression request.
    size_t staging_buffer_size{0};
    for (size_t partition_idx : partition_indexes) {
      staging_buffer_size +=
        rmm::align_up(host_source_buffer.get_partition(partition_idx).size(), size_t{8});
    }

    staging_buffers.emplace_back(staging_buffer_size, stream, mr);
    auto* device_staging_buffer = &staging_buffers.back();
    auto* staging_ptr           = reinterpret_cast<std::byte*>(device_staging_buffer->data());
    c_batch.reserve(c_batch.size() + partition_indexes.size());
    for (size_t partition_idx : partition_indexes) {
      auto partition = host_source_buffer.get_partition(partition_idx);
      c_batch.add(
        staging_ptr, reinterpret_cast<std::byte const*>(partition.data()), partition.size());
      staging_ptr += rmm::align_up(partition.size(), size_t{8});
    }

    d_batch.add(host_source_buffer,
                compression_config,
                std::vector<size_t>(partition_indexes.begin(), partition_indexes.end()),
                device_staging_buffer,
                output_ptr);
    return;
  }

  // Each selected uncompressed partition is copied directly into the output buffer.
  c_batch.reserve(c_batch.size() + partition_indexes.size());

  auto* current_output_ptr = output_ptr;
  for (size_t partition_idx : partition_indexes) {
    auto partition = host_source_buffer.get_partition(partition_idx);
    c_batch.add(
      current_output_ptr, reinterpret_cast<std::byte const*>(partition.data()), partition.size());
    current_output_ptr += partition.size();
  }
}

}  // namespace

template <typename T>
rmm::device_buffer contiguous_output_column_builder<T>::allocate_output_buffer(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  return make_output_buffer(_num_rows, cudf::size_of(_cudf_type), stream, mr);
}

template <typename T>
void contiguous_output_column_builder<T>::submit_materialization_requests(
  utility::copy_batch& c_batch,
  decompression_batch&,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  size_t num_copy_requests{0};
  for (const auto& [/* row_group */ _, pruning_result] : *_pruning_results) {
    num_copy_requests += pruning_result.consolidated_partitions().size();
  }
  c_batch.reserve(c_batch.size() + num_copy_requests);

  _output_buffer        = allocate_output_buffer(stream, mr);
  std::byte* target_ptr = reinterpret_cast<std::byte*>(_output_buffer.data());
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const T& column        = get_column<T>(row_group, _column_idx);
    const auto column_view = column.view();
    auto* source_ptr = reinterpret_cast<std::byte const*>(column_view.template data<std::byte>());
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      const size_t size_in_bytes = (partition.end - partition.start) * cudf::size_of(_cudf_type);
      auto const* partition_source_ptr = source_ptr + partition.start * cudf::size_of(_cudf_type);
      c_batch.add(target_ptr, partition_source_ptr, size_in_bytes);
      target_ptr += size_in_bytes;
    }
  }
}

template <typename T>
std::unique_ptr<cudf::column> contiguous_output_column_builder<T>::make_cudf_column(
  rmm::cuda_stream_view /* concatenation_stream */, rmm::device_async_resource_ref /* mr */)
{
  cudf::size_type null_count = 0;
  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(_output_buffer), rmm::device_buffer(), null_count);
}

rmm::device_buffer compressed_sliced_output_column_builder::allocate_output_buffer(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  return make_output_buffer(_num_rows, cudf::size_of(_cudf_type), stream, mr);
}

void compressed_sliced_output_column_builder::submit_materialization_requests(
  utility::copy_batch& c_batch,
  decompression_batch& d_batch,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  _output_buffer        = allocate_output_buffer(stream, mr);
  std::byte* target_ptr = reinterpret_cast<std::byte*>(_output_buffer.data());
  size_t target_ptr_offset{0};
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    if (pruning_result.partition_size() <= 0) {
      throw std::invalid_argument(
        "Compressed-sliced materialization requires zone_map_partition_size > 0.");
    }
    const auto& column     = get_column<compressed_sliced_column_interface>(row_group, _column_idx);
    auto const data_buffer = column.data_buffer_view();
    auto const& compression_config = column.get_compression_config();
    auto const& partition_indexes  = pruning_result.partition_indexes();
    auto* row_group_target_ptr     = target_ptr + target_ptr_offset;

    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _staging_buffers,
                                           data_buffer,
                                           compression_config,
                                           partition_indexes,
                                           row_group_target_ptr,
                                           stream,
                                           mr);

    target_ptr_offset += pruning_result.num_rows() * cudf::size_of(_cudf_type);
  }
}

std::unique_ptr<cudf::column> compressed_sliced_output_column_builder::make_cudf_column(
  rmm::cuda_stream_view /* concatenation_stream */, rmm::device_async_resource_ref /* mr */)
{
  cudf::size_type null_count = 0;
  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(_output_buffer), rmm::device_buffer(), null_count);
}

template <bool large_string_mode>
std::pair<rmm::device_buffer, rmm::device_buffer>
string_compressed_sliced_output_column_builder<large_string_mode>::allocate_output_buffers(
  size_t char_buffer_size,
  size_t offset_buffer_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return {make_output_buffer(char_buffer_size, stream, mr),
          make_output_buffer(offset_buffer_size, stream, mr)};
}

template <bool large_string_mode>
void string_compressed_sliced_output_column_builder<
  large_string_mode>::submit_materialization_requests(utility::copy_batch& c_batch,
                                                      decompression_batch& d_batch,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  size_t char_buffer_size{0};
  size_t offset_buffer_size = sizeof(
    typename string_compressed_sliced_output_column_builder<large_string_mode>::offsets_type);
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    if (pruning_result.partition_size() <= 0) {
      throw std::invalid_argument(
        "Compressed-sliced string materialization requires zone_map_partition_size > 0.");
    }
    const auto& column =
      get_column<string_compressed_sliced_column_interface>(row_group, _column_idx);
    for (size_t partition_idx : pruning_result.partition_indexes()) {
      char_buffer_size += column.get_char_partition_output_size(partition_idx);
    }
    offset_buffer_size +=
      pruning_result.num_rows() *
      sizeof(
        typename string_compressed_sliced_output_column_builder<large_string_mode>::offsets_type);
  }

  std::tie(_char_buffer, _offset_buffer) =
    allocate_output_buffers(char_buffer_size, offset_buffer_size, stream, mr);
  auto* char_target_ptr   = reinterpret_cast<std::byte*>(_char_buffer.data());
  auto* offset_target_ptr = reinterpret_cast<std::byte*>(_offset_buffer.data());

  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const auto& column =
      get_column<string_compressed_sliced_column_interface>(row_group, _column_idx);
    auto const char_buffer         = column.char_buffer_view();
    auto const offset_buffer       = column.offset_buffer_view();
    auto const& compression_config = column.get_compression_config();
    auto const& partition_indexes  = pruning_result.partition_indexes();

    size_t char_output_size{0};
    for (size_t partition_idx : partition_indexes) {
      char_output_size += column.get_char_partition_output_size(partition_idx);
    }

    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _char_staging_buffers,
                                           char_buffer,
                                           compression_config,
                                           partition_indexes,
                                           char_target_ptr,
                                           stream,
                                           mr);
    char_target_ptr += char_output_size;

    const size_t offset_output_size =
      pruning_result.num_rows() *
      sizeof(
        typename string_compressed_sliced_output_column_builder<large_string_mode>::offsets_type);
    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _offset_staging_buffers,
                                           offset_buffer,
                                           compression_config,
                                           partition_indexes,
                                           offset_target_ptr,
                                           stream,
                                           mr);
    offset_target_ptr += offset_output_size;
  }
}

template <bool large_string_mode>
std::unique_ptr<cudf::column>
string_compressed_sliced_output_column_builder<large_string_mode>::make_cudf_column(
  rmm::cuda_stream_view concatenation_stream, rmm::device_async_resource_ref /* mr */)
{
  // Adjust offsets
  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  size_t num_partitions     = 0;
  for (auto const& [/* row_group */ _, pruning_result] : *_pruning_results) {
    num_partitions += pruning_result.partition_indexes().size();
  }

  rmm::device_uvector<offsets_type> partition_char_offsets_buffer(
    num_partitions, concatenation_stream, cudf_pinned_resource);
  auto* partition_char_offsets = partition_char_offsets_buffer.data();
  rmm::device_uvector<cudf::size_type> partition_row_offsets_buffer(
    num_partitions, concatenation_stream, cudf_pinned_resource);
  auto* partition_row_offsets = partition_row_offsets_buffer.data();

  concatenation_stream.synchronize();

  offsets_type char_offset    = 0;
  cudf::size_type row_offset  = 0;
  size_t partition_offset_idx = 0;
  auto fill_partition_offsets = [&](const string_compressed_sliced_column_interface& column,
                                    const pruning_result_t& pruning_result,
                                    size_t partition_offset_idx) {
    auto chars_partition_offset_idx = partition_offset_idx;
    for (auto partition_idx : pruning_result.partition_indexes()) {
#ifndef NDEBUG
      GQE_LOG_DEBUG("partition_idx = {}, char_offset = {}", partition_idx, char_offset);
#endif
      auto const partition_char_size = column.get_char_partition_output_size(partition_idx);
      auto const max_char_offset = static_cast<size_t>(std::numeric_limits<offsets_type>::max());
      auto const current_char_offset = static_cast<size_t>(char_offset);
      if (partition_char_size > max_char_offset ||
          current_char_offset > max_char_offset - partition_char_size) {
        throw std::overflow_error("Compressed-sliced string character offset exceeds offset type");
      }

      partition_char_offsets[chars_partition_offset_idx] = char_offset;
      char_offset += static_cast<offsets_type>(partition_char_size);
      ++chars_partition_offset_idx;
    }

    auto const partition_size     = pruning_result.partition_size();
    auto row_partition_offset_idx = partition_offset_idx;
    for (auto const& consolidated_partition : pruning_result.consolidated_partitions()) {
      auto partition_start = consolidated_partition.start;
      while (partition_start < consolidated_partition.end) {
        auto const partition_end =
          std::min<cudf::size_type>(partition_start + partition_size, consolidated_partition.end);
        auto const row_count = partition_end - partition_start;
#ifndef NDEBUG
        auto const partition_idx = static_cast<size_t>(partition_start / partition_size);
        GQE_LOG_DEBUG("partition_idx = {}, row_offset = {}, char_offset = {}",
                      partition_idx,
                      row_offset,
                      char_offset);
#endif
        partition_row_offsets[row_partition_offset_idx] = row_offset;
        row_offset += row_count;
        ++row_partition_offset_idx;
        partition_start = partition_end;
      }
    }
  };

  for (auto& [row_group, pruning_result] : *_pruning_results) {
    auto& column = get_column<string_compressed_sliced_column_interface>(row_group, _column_idx);
    fill_partition_offsets(column, pruning_result, partition_offset_idx);
    partition_offset_idx += pruning_result.partition_indexes().size();
  }

  // create a copy on device before adjust_offsets since we will do a binary search over this buffer
  rmm::device_uvector<cudf::size_type> d_partition_row_offsets_buffer(partition_row_offsets_buffer,
                                                                      concatenation_stream);
  auto d_partition_row_offsets = d_partition_row_offsets_buffer.data();

  adjust_offsets_api(reinterpret_cast<offsets_type*>(_offset_buffer.data()),
                     _num_rows,
                     num_partitions,
                     d_partition_row_offsets,
                     partition_char_offsets,
                     _char_buffer.size(),
                     concatenation_stream);
  GQE_LOG_DEBUG(
    "Adjusted offsets; _column_idx = {}, num_partitions = {}, char_offset = {}, row_offset = {}, "
    "_num_rows = {}, _char_buffer.size() = {}",
    _column_idx,
    num_partitions,
    char_offset,
    row_offset,
    _num_rows,
    _char_buffer.size());

  // Create CUDF column
  auto offset_column = std::make_unique<cudf::column>(
    offset_element_type, _num_rows + 1, std::move(_offset_buffer), rmm::device_buffer(), 0);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(1);
  children.push_back(std::move(offset_column));
  return std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::STRING),
                                        _num_rows,
                                        std::move(_char_buffer),
                                        rmm::device_buffer(),
                                        0,
                                        std::move(children));
}

template class string_compressed_sliced_output_column_builder<false>;
template class string_compressed_sliced_output_column_builder<true>;

output_table_builder::output_table_builder(context_reference ctx_ref,
                                           std::vector<cudf::size_type> column_indexes,
                                           std::vector<cudf::data_type> data_types,
                                           std::unique_ptr<pruning_results_t> pruning_results)
  : _ctx_ref(ctx_ref),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _pruning_results(std::shared_ptr<pruning_results_t>(std::move(pruning_results)))
{
}

std::unique_ptr<cudf::table> output_table_builder::build(rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  // Select one builder per projected column. The first row group is only used to inspect storage
  // details that must be consistent across row groups, such as string offset width.
  _output_columns.clear();
  _output_columns.reserve(_column_indexes.size());

  const auto* representative_row_group = _pruning_results->front().first;
  for (size_t i = 0; i < _column_indexes.size(); ++i) {
    const auto column_idx             = _column_indexes[i];
    const auto cudf_type              = _data_types[i];
    const auto& representative_column = representative_row_group->get_column(column_idx);
    _output_columns.push_back(make_output_column_builder(cudf_type,
                                                         representative_column.layout(),
                                                         _pruning_results,
                                                         column_idx,
                                                         representative_row_group));
  }

  std::unique_lock<std::mutex> overlap_mtx_lock;
  auto copy_stream          = stream;
  auto decompression_stream = stream;
  auto column_build_stream  = stream;

  auto& shared_stream = _ctx_ref._task_manager_context->copy_engine_stream;
  if (_ctx_ref._query_context->parameters.use_overlap_mtx) {
    overlap_mtx_lock     = std::unique_lock{*shared_stream.mtx};
    decompression_stream = shared_stream.stream;
  }

  // Prepare all materialization work first. Builders allocate their output/staging buffers here
  // and enqueue copy/decompression requests, but do not create cuDF columns yet.
  utility::copy_batch c_batch;
  decompression_batch d_batch;
  for (auto& output_column : _output_columns) {
    output_column->submit_materialization_requests(c_batch, d_batch, copy_stream, mr);
  }

  const bool has_compressed_buffers = !d_batch.empty();
  if (_ctx_ref._query_context->parameters.use_overlap_mtx && !has_compressed_buffers) {
    // For contiguous columns, run cudf::concatenate on the shared stream since we do not have
    // batched memcpy for this case.
    column_build_stream = shared_stream.stream;
  }

  // Copy selected contiguous data and compressed partition bytes into their destination/staging
  // buffers. Decompression, if needed, consumes those staging buffers afterward.
  if (!c_batch.empty()) {
    utility::semaphore_acquire_guard guard(
      _ctx_ref._task_manager_context->batched_memcpy_semaphore);
    c_batch.execute(copy_stream,
                    _ctx_ref._query_context->parameters.in_memory_dummy_copy_multiplier);
    copy_stream.synchronize();
  }

  if (has_compressed_buffers) {
    utility::nvtx_scoped_range nvtx_range("Decompressing columns");
    decompression_stream.synchronize();
    utility::semaphore_acquire_guard guard(_ctx_ref._task_manager_context->decompress_semaphore);
    d_batch.execute(decompression_stream);
    decompression_stream.synchronize();
  }

  auto cudf_column_view =
    _output_columns | std::views::transform([column_build_stream, mr](auto& output_column) {
      utility::nvtx_scoped_range nvtx_range("Creating cuDF column");
      return output_column->make_cudf_column(column_build_stream, mr);
    });
  std::vector<std::unique_ptr<cudf::column>> cudf_columns(cudf_column_view.begin(),
                                                          cudf_column_view.end());

  cudaEvent_t columns_ready_evt{};
  GQE_CUDA_TRY(cudaEventCreateWithFlags(&columns_ready_evt, cudaEventDisableTiming));
  GQE_CUDA_TRY(cudaEventRecord(columns_ready_evt, column_build_stream.value()));
  GQE_CUDA_TRY(cudaStreamWaitEvent(stream.value(), columns_ready_evt));

  if (_ctx_ref._query_context->parameters.use_overlap_mtx) {
    shared_stream.stream.synchronize();
    overlap_mtx_lock.unlock();
  }
  GQE_CUDA_TRY(cudaEventDestroy(columns_ready_evt));

  return std::make_unique<cudf::table>(std::move(cudf_columns));
}

in_memory_read_task::in_memory_read_task(context_reference ctx_ref,
                                         int32_t task_id,
                                         int32_t stage_id,
                                         std::vector<const row_group_interface*> row_groups,
                                         std::vector<cudf::size_type> column_indexes,
                                         std::vector<cudf::data_type> data_types,
                                         memory_kind::type memory_kind,
                                         std::optional<arrow::compute::Expression> partial_filter,
                                         std::vector<std::shared_ptr<task>> subquery_tasks,
                                         bool force_zero_copy_disable)
  : read_task_base(ctx_ref, task_id, stage_id, std::move(subquery_tasks)),
    _row_groups(std::move(row_groups)),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _memory_kind(std::move(memory_kind)),
    _partial_filter(std::move(partial_filter)),
    _force_zero_copy_disable(force_zero_copy_disable)
{
}

std::unique_ptr<pruning_results_t> detail::evaluate_in_memory_pruning(
  std::span<row_group_interface const* const> row_groups,
  std::span<cudf::size_type const> column_indexes,
  std::optional<arrow::compute::Expression> const& partial_filter,
  bool use_partition_pruning,
  cudf::size_type partition_size)
{
  GQE_EXPECTS(partition_size >= 0, "partition_size must be non-negative");

  auto pruning_results = std::make_unique<pruning_results_t>();
  if (partial_filter && partition_size != 0 && use_partition_pruning) {
    pruning_results->reserve(row_groups.size());
    std::for_each(row_groups.begin(),
                  row_groups.end(),
                  [&pruning_results, &partial_filter, partition_size](auto const* rg) {
                    if (!rg->zone_map()) {
                      throw std::logic_error("Row group should have a zone map but none exists");
                    }
                    auto const partitions = rg->zone_map()->evaluate(*partial_filter);
                    const auto has_unpruned_partitions =
                      std::find_if(partitions.begin(), partitions.end(), [](const auto& partition) {
                        return !partition.pruned;
                      }) != partitions.end();
                    if (has_unpruned_partitions) {
                      // TODO Change output of evaluate to pruning_result?
                      pruning_results->emplace_back(rg,
                                                    pruning_result_t{partitions, partition_size});
                    }
                  });
    GQE_LOG_DEBUG("Row groups after pruning: {}", pruning_results->size());
    return pruning_results;
  } else {
    GQE_LOG_DEBUG("No partial filter; processing all row groups");
    pruning_results->reserve(row_groups.size());
    std::transform(
      row_groups.begin(),
      row_groups.end(),
      std::back_inserter(*pruning_results),
      [column_indexes, partition_size](auto const* rg) {
        std::vector<cudf::size_type> null_counts(rg->num_columns(), 0);
        std::for_each(column_indexes.begin(), column_indexes.end(), [&null_counts, &rg](auto i) {
          null_counts[i] = rg->get_column(i).null_count();
        });
        const zone_map::partition partition{.pruned      = false,
                                            .start       = 0,
                                            .end         = static_cast<cudf::size_type>(rg->size()),
                                            .null_counts = null_counts};
        // When zone map partition size is disabled
        // (<= 0), compressed_sliced columns use one slice of row-group size. Avoid _partition_size
        // == 0 in pruning_result_t
        const auto const_row_group_rows = static_cast<cudf::size_type>(rg->size());
        const cudf::size_type effective_partition_size =
          partition_size > 0 ? partition_size
                             : std::max(static_cast<cudf::size_type>(1), const_row_group_rows);
        return row_group_with_pruning_result_t{
          rg, pruning_result_t{{partition}, effective_partition_size}};
      });
    return pruning_results;
  }
}

std::unique_ptr<pruning_results_t> in_memory_read_task::evaluate_partial_filter()
{
  return detail::evaluate_in_memory_pruning(
    _row_groups,
    _column_indexes,
    _partial_filter,
    get_query_context()->parameters.use_partition_pruning,
    get_query_context()->parameters.zone_map_partition_size);
}

bool in_memory_read_task::is_zero_copy_possible(const pruning_results_t& pruning_results) const
{
  // The function emit_zero_copy_result assumes the positive outcome of the checks below.
  if (!get_query_context()->parameters.read_use_zero_copy) {
    GQE_LOG_DEBUG("Zero-copy read is disabled by the query context parameters");
    return false;
  }
  if (!memory_kind::is_gpu_accessible(_memory_kind)) {
    GQE_LOG_WARN(
      "Zero-copy read is not possible because the GPU cannot access pageable memory on this "
      "system");
    return false;
  }
  if (pruning_results.size() > 1) {
    GQE_LOG_DEBUG(
      "Zero-copy read is not possible because there are multiple row groups with "
      "candidate partitions");
    return false;
  }
  if (pruning_results.size() == 0) {
    throw std::logic_error("There should be at least a single candidate partition");
  }
  auto& [row_group, pruning_result] = pruning_results.front();
  for (auto column_idx : _column_indexes) {
    auto layout = row_group->get_column(column_idx).layout();
    if (layout != column_layout::contiguous) {
      GQE_LOG_DEBUG(
        "Zero-copy read is not possible because at least one column is not contiguous: "
        "layout={}",
        layout);
      return false;
    }
  }
  auto num_partitions = pruning_result.consolidated_partitions().size();
  if (num_partitions > 1) {
    GQE_LOG_DEBUG(
      "Zero-copy read is not possible because there are multiple discontiguous "
      "candidate partitions: num_partitions={}",
      num_partitions);
    return false;
  }
  if (num_partitions != 1) {
    throw std::logic_error("There should be exactly one contiguous candidate partition");
  }
  return true;
}

void in_memory_read_task::emit_empty_table()
{
  GQE_LOG_DEBUG("Returning empty table");
  std::vector<std::unique_ptr<cudf::column>> empty_columns;
  empty_columns.reserve(_data_types.size());
  for (const auto& data_type : _data_types) {
    // cudf::make_empty_column will fail for nested types, e.g., lists and structs. In this
    // case, we have to use cudf::empty_like on an existing column, or reproduce it.
    // TODO There is no test of the entire query execution pipeline when an empty table is
    // returned
    auto empty_column = cudf::make_empty_column(data_type);
    empty_columns.push_back(std::move(empty_column));
  }
  auto empty_table = std::make_unique<cudf::table>(std::move(empty_columns));
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=empty_table, "
    "output_size={}.",
    task_id(),
    stage_id(),
    empty_table->num_rows());
  emit_result(std::move(empty_table));
}

// This method throws std::logic_error if certain preconditions are not met. These are checked
// by is_zero_copy_possible. They are duplicated here to ensure that the code doesn't break if
// is_zero_copy_possible is changed.
void in_memory_read_task::emit_zero_copy_result(std::unique_ptr<pruning_results_t> pruning_results)
{
  GQE_LOG_DEBUG("Performing zero-copy read");

  if (pruning_results->size() != 1) {
    throw std::logic_error("There must be a single row group with candidate partitions");
  }
  const auto& [row_group, pruning_result] = pruning_results->front();

  if (pruning_result.consolidated_partitions().size() != 1) {
    throw std::logic_error("There must be a single consolidated partition");
  }
  const auto& partition = pruning_result.consolidated_partitions().front();

  std::vector<cudf::column_view> columns;
  columns.reserve(_column_indexes.size());
  for (auto column_idx : _column_indexes) {
    auto& column = row_group->get_column(column_idx);
    if (column.layout() != column_layout::contiguous) {
      throw std::logic_error("The column must be contiguous");
    }
    auto const full_column_view        = dynamic_cast<contiguous_column_interface&>(column).view();
    const auto size                    = partition.end - partition.start;
    const auto offset                  = partition.start;
    const auto null_counts             = partition.null_counts[column_idx];
    const auto partitioned_column_view = slice_column(full_column_view, size, offset, null_counts);
    columns.push_back(partitioned_column_view);
  }
  cudf::table_view result(columns);
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_reference, "
    "output_size={}.",
    task_id(),
    stage_id(),
    result.num_rows());
  emit_result(result);
}

void in_memory_read_task::emit_copied_result(std::unique_ptr<pruning_results_t> pruning_results)
{
  GQE_LOG_DEBUG("Copying data to GPU");
  output_table_builder builder(
    get_context_reference(), _column_indexes, _data_types, std::move(pruning_results));
  auto result =
    builder.build(cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_value, output_size={}.",
    task_id(),
    stage_id(),
    result->num_rows());
  emit_result(std::move(result));
}

void in_memory_read_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range in_memory_read_task_range("in_memory_read_task");

  auto pruning_results = evaluate_partial_filter();

  if (_column_indexes.empty()) {
    // No columns projected (e.g. SELECT COUNT(*) FROM t). A 0-column table has
    // num_rows()==0 in cudf, so we must carry the row count via a phantom EMPTY column.
    // Using EMPTY avoids allocating ~1 byte/row (INT8 would); the public cudf::column
    // constructor is the supported way to size an EMPTY column with a zero-size data buffer.
    cudf::size_type total_rows = 0;
    for (auto const& [rg, pr] : *pruning_results)
      total_rows += pr.num_rows();
    std::vector<std::unique_ptr<cudf::column>> phantom;
    phantom.push_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::EMPTY},
                                                     total_rows,
                                                     rmm::device_buffer{},
                                                     rmm::device_buffer{},
                                                     0));
    emit_result(std::make_unique<cudf::table>(std::move(phantom)));
  } else if (pruning_results->empty()) {
    emit_empty_table();
  } else if (is_zero_copy_possible(*pruning_results)) {
    emit_zero_copy_result(std::move(pruning_results));
  } else {
    emit_copied_result(std::move(pruning_results));
  }

  GQE_LOG_DEBUG("Finishing read task");
  remove_dependencies();
  GQE_LOG_DEBUG("Finished read task");
}

namespace {

/**
 * @brief Select the physical storage layout for a column.
 *
 * Uncompressed columns use contiguous storage. Compressed strings select their offset width based
 * on the size of the character data; all other compressed columns use compressed-sliced storage.
 *
 * @param column Column whose layout is being selected.
 * @param configured_format Configured primary compression format.
 * @param stream Stream used when inspecting string character data.
 * @return The storage layout to use when materializing the column.
 */
column_layout pick_column_layout(cudf::column_view const& column,
                                 compression_format configured_format,
                                 rmm::cuda_stream_view stream)
{
  if (configured_format == compression_format::none) { return column_layout::contiguous; }
  if (column.type().id() != cudf::type_id::STRING) { return column_layout::compressed_sliced; }

  auto const chars_size = cudf::strings_column_view(column).chars_size(stream);
  return chars_size > std::numeric_limits<int32_t>::max()
           ? column_layout::large_string_compressed_sliced
           : column_layout::string_compressed_sliced;
}

/**
 * @brief Materialize and append one process-local row group.
 */
std::optional<table_statistics> write_row_group(
  in_memory_table::row_group_appender& row_group_appender,
  cudf::table_view input_table,
  rmm::device_async_resource_ref memory_resource,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> const& column_indexes,
  std::vector<cudf::data_type> const& data_types,
  compression::configuration const& compression_config,
  cudf::size_type partition_size,
  rmm::cuda_stream_view stream)
{
  table_statistics row_group_statistics(input_table.num_rows(), data_types);
  std::vector<std::unique_ptr<column_base>> columns(column_indexes.size());
  for (cudf::size_type column_idx = 0; column_idx < input_table.num_columns(); ++column_idx) {
    auto const& input_column   = input_table.column(column_idx);
    auto const destination_idx = column_indexes.at(column_idx);
    auto cudf_column           = cudf::column(input_column, stream, memory_resource);

    switch (
      pick_column_layout(input_column, compression_config.primary_compression_format, stream)) {
      case column_layout::contiguous:
        GQE_LOG_TRACE("Uncompressed column size {}", cudf_column.size());
        columns.at(destination_idx) =
          std::make_unique<contiguous_column>(std::move(cudf_column), stream, memory_resource);
        break;
      case column_layout::compressed_sliced:
        GQE_LOG_TRACE("Compressed sliced column size {}", cudf_column.size());
        columns.at(destination_idx) =
          std::make_unique<compressed_sliced_column>(std::move(cudf_column),
                                                     partition_size,
                                                     memory_kind,
                                                     compression_config,
                                                     stream,
                                                     memory_resource);
        break;
      case column_layout::string_compressed_sliced:
        columns.at(destination_idx) =
          std::make_unique<string_compressed_sliced_column<false>>(std::move(cudf_column),
                                                                   partition_size,
                                                                   memory_kind,
                                                                   compression_config,
                                                                   stream,
                                                                   memory_resource);
        break;
      case column_layout::large_string_compressed_sliced:
        columns.at(destination_idx) =
          std::make_unique<string_compressed_sliced_column<true>>(std::move(cudf_column),
                                                                  partition_size,
                                                                  memory_kind,
                                                                  compression_config,
                                                                  stream,
                                                                  memory_resource);
        break;
    }

    row_group_statistics.add_column_statistics(
      {.column_id         = static_cast<size_t>(destination_idx),
       .compression_stats = columns.at(destination_idx)->get_compression_stats()});
  }

  auto zone_map = partition_size > 0
                    ? std::make_unique<gqe::local_zone_map>(input_table, partition_size)
                    : nullptr;
  row_group_appender(
    storage::row_group(input_table.num_rows(), std::move(columns), std::move(zone_map)));

  return row_group_statistics;
}

/**
 * @brief Materialize and append one shared-memory row group.
 *
 * @return Statistics for a published row group, or `std::nullopt` when this rank does not publish.
 */
std::optional<table_statistics> write_row_group(
  shared_in_memory_table::row_group_appender& row_group_appender,
  cudf::table_view input_table,
  rmm::device_async_resource_ref memory_resource,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> const& column_indexes,
  std::vector<cudf::data_type> const& data_types,
  compression::configuration const& compression_config,
  cudf::size_type partition_size,
  rmm::cuda_stream_view stream)
{
  // TODO: Fix multi-rank write statistics before allowing every rank to publish.
  if (!gqe::utility::multi_process::nvshmem_rank_zero()) { return std::nullopt; }

  table_statistics row_group_statistics(input_table.num_rows(), data_types);
  auto& segment = row_group_appender.segment();
  shared_memory_allocator alloc(segment.get_segment_manager());
  shared_row_group_storage shared_row_group(input_table.num_rows(), alloc);
  shared_row_group.columns.reserve(column_indexes.size());
  std::vector<std::optional<column_storage_handle>> column_handles(column_indexes.size());
  for (cudf::size_type column_idx = 0; column_idx < input_table.num_columns(); ++column_idx) {
    auto const& input_column   = input_table.column(column_idx);
    auto const destination_idx = column_indexes.at(column_idx);
    auto const layout =
      pick_column_layout(input_column, compression_config.primary_compression_format, stream);
    column_compression_statistics compression_statistics;

    switch (layout) {
      case column_layout::contiguous: {
        using storage_t = shared_contiguous_column_storage;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          input_column, stream, memory_resource, alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = shared_contiguous_column(segment, handle).get_compression_stats();
        break;
      }
      case column_layout::compressed_sliced: {
        using storage_t = shared_compressed_sliced_column_storage;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          cudf::column(input_column, stream, memory_resource),
          partition_size,
          memory_kind,
          compression_config,
          stream,
          memory_resource,
          alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = storage->get_compression_stats();
        break;
      }
      case column_layout::string_compressed_sliced: {
        using storage_t = shared_string_compressed_sliced_column_storage<false>;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          cudf::column(input_column, stream, memory_resource),
          partition_size,
          memory_kind,
          compression_config,
          stream,
          memory_resource,
          alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = storage->get_compression_stats();
        break;
      }
      case column_layout::large_string_compressed_sliced: {
        using storage_t = shared_string_compressed_sliced_column_storage<true>;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          cudf::column(input_column, stream, memory_resource),
          partition_size,
          memory_kind,
          compression_config,
          stream,
          memory_resource,
          alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = storage->get_compression_stats();
        break;
      }
    }

    row_group_statistics.add_column_statistics(
      {.column_id         = static_cast<size_t>(destination_idx),
       .compression_stats = std::move(compression_statistics)});
  }

  for (auto const& handle : column_handles) {
    if (!handle) { throw std::logic_error("Shared write did not populate every table column"); }
    shared_row_group.columns.push_back(*handle);
  }

  if (partition_size > 0) {
    shared_memory_allocator zone_map_alloc(segment.get_segment_manager());
    auto* zone_map_storage = segment.construct<gqe::shared_zone_map_storage>(
      boost::interprocess::anonymous_instance)(input_table,
                                               partition_size,
                                               stream,
                                               rmm::mr::get_current_device_resource_ref(),
                                               memory_resource,
                                               zone_map_alloc);
    shared_row_group.zone_map = segment.get_handle_from_address(zone_map_storage);
  }

  // Ensure all asynchronous copies into the shared segment have completed before publishing it.
  stream.synchronize();
  row_group_appender(std::move(shared_row_group));

  return row_group_statistics;
}

}  // namespace

in_memory_write_task::in_memory_write_task(context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::shared_ptr<task> input,
                                           rmm::device_async_resource_ref non_owned_memory_resource,
                                           row_group_appender_t row_group_appender,
                                           memory_kind::type memory_kind,
                                           std::vector<cudf::size_type> column_indexes,
                                           std::vector<cudf::data_type> data_types,
                                           table_statistics_manager* statistics)
  : write_task_base(ctx_ref, task_id, stage_id, input),
    _non_owned_memory_resource(non_owned_memory_resource),
    _row_group_appender(std::move(row_group_appender)),
    _memory_kind(memory_kind),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _statistics(statistics)
{
}

void in_memory_write_task::execute()
{
  GQE_LOG_TRACE("Write task execute");
  prepare_dependencies();
  utility::nvtx_scoped_range in_memory_write_task_range("in_memory_write_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto const input_table        = *dependent_tasks[0]->result();
  auto const& parameters        = get_query_context()->parameters;
  auto const partition_size     = parameters.zone_map_partition_size;
  auto const compression_config = compression::configuration{
    .primary_compression_format            = parameters.compression_format,
    .secondary_compression_format          = parameters.secondary_compression_format,
    .compression_chunk_size                = parameters.compression_chunk_size,
    .compression_ratio_threshold           = parameters.compression_ratio_threshold,
    .secondary_compression_ratio_threshold = parameters.secondary_compression_ratio_threshold,
    .secondary_compression_multiplier_threshold =
      parameters.secondary_compression_multiplier_threshold,
    .use_cpu_compression = parameters.use_cpu_compression,
    .compression_level   = parameters.compression_level,
    .decompress_backend  = parameters.decompression_backend};
  if (compression_config.primary_compression_format != compression_format::none &&
      partition_size <= 0) {
    throw std::invalid_argument(
      "Compressed in-memory writes require zone_map_partition_size > 0 because "
      "compressed_sliced_column is the only supported compressed storage layout.");
  }

  auto const num_columns = _column_indexes.size();
  if (_data_types.size() != num_columns) {
    throw std::length_error("Data types must have the same length as the number of columns");
  }
  if (static_cast<decltype(num_columns)>(input_table.num_columns()) != num_columns) {
    throw std::invalid_argument("Query result schema must match Parquet output schema");
  }
  for (std::size_t column_idx = 0; column_idx < num_columns; ++column_idx) {
    if (_data_types[column_idx] != input_table.column(column_idx).type()) {
      throw std::invalid_argument("Query result schema must match Parquet output schema");
    }
  }

  auto const stream = cudf::get_default_stream();

  auto row_group_statistics = std::visit(
    [&](auto& row_group_appender) {
      return write_row_group(row_group_appender,
                             input_table,
                             _non_owned_memory_resource,
                             _memory_kind,
                             _column_indexes,
                             _data_types,
                             compression_config,
                             partition_size,
                             stream);
    },
    _row_group_appender);
  if (row_group_statistics) { _statistics->append_table_statistics(*row_group_statistics); }

  GQE_LOG_TRACE("Execute in-memory write task: task_id={}, stage_id={}, input_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows());
  remove_dependencies();
}

in_memory_readable_view::in_memory_readable_view(in_memory_table_interface* non_owning_table)
  : readable_view(), _non_owning_table(non_owning_table)
{
}

std::vector<std::unique_ptr<read_task_base>> in_memory_readable_view::get_read_tasks(
  std::vector<readable_view::task_parameters>&& task_parameters,
  context_reference ctx_ref,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types)
{
  assert(!task_parameters.empty() && "Must have at least one read task");
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0 && "Stage ID must be a positive value");

  assert(task_parameters.size() <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  auto parallelism = static_cast<int64_t>(task_parameters.size());

  // Transform column names into column indexes
  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(column_names.cbegin(),
                 column_names.cend(),
                 std::back_inserter(column_indexes),
                 [this](const std::string& column_name) {
                   return _non_owning_table->get_column_index(column_name);
                 });

  // Acquire shared read latch on row groups and release on function return
  std::shared_lock latch_guard(_non_owning_table->row_group_latch());
  auto const nrow_groups_unsigned = _non_owning_table->num_row_groups();

  assert(nrow_groups_unsigned <=
         static_cast<decltype(nrow_groups_unsigned)>(std::numeric_limits<int64_t>::max()));
  const auto nrow_groups = static_cast<int64_t>(nrow_groups_unsigned);

  const auto max_nrow_groups_per_instance = utility::divide_round_up(nrow_groups, parallelism);

  // Transform partial filter to zone map filter. Assumes that the partial filter is the same for
  // all tasks.
  const auto zone_map_filter =
    transform_partial_filter(task_parameters.front().partial_filter.get());

  // Create tasks
  std::vector<std::unique_ptr<read_task_base>> read_tasks;
  read_tasks.reserve(task_parameters.size());
  {
    // Iterate over multiple iterators at the same time
    auto task            = task_parameters.begin();
    int64_t begin_offset = 0;

    while (task != task_parameters.end() && begin_offset < nrow_groups) {
      const auto end_offset = std::min(begin_offset + max_nrow_groups_per_instance, nrow_groups);

      // Get a chunk of row groups
      std::vector<const row_group_interface*> row_groups_chunk;
      assert(end_offset >= begin_offset);
      row_groups_chunk.reserve(end_offset - begin_offset);
      for (auto row_group_idx = begin_offset; row_group_idx < end_offset; ++row_group_idx) {
        row_groups_chunk.push_back(
          _non_owning_table->row_group_at(static_cast<std::size_t>(row_group_idx)));
      }

      // Create a new read task
      auto read_task = std::make_unique<in_memory_read_task>(ctx_ref,
                                                             task->task_id,
                                                             stage_id,
                                                             std::move(row_groups_chunk),
                                                             column_indexes,
                                                             data_types,
                                                             _non_owning_table->get_memory_kind(),
                                                             zone_map_filter,
                                                             std::move(task->subquery_tasks));
      read_tasks.push_back(std::move(read_task));

      // Advance the iterators
      ++task, begin_offset += max_nrow_groups_per_instance;
    }
  }

  return read_tasks;
}

std::unique_ptr<qep::iterate_task> in_memory_readable_view::get_iterate_task(
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  std::optional<cudf::size_type> partition_column_idx,
  std::unique_ptr<gqe::expression> partial_filter)
{
  GQE_EXPECTS(data_types.size() == column_names.size(),
              "data_types must have the same length as the number of columns",
              std::length_error);
  GQE_EXPECTS(!partition_column_idx.has_value(),
              "in-memory QEP reads do not support partition-column pruning",
              std::logic_error);

  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(column_names.cbegin(),
                 column_names.cend(),
                 std::back_inserter(column_indexes),
                 [this](std::string const& column_name) {
                   return _non_owning_table->get_column_index(column_name);
                 });

  std::vector<row_group_interface const*> row_groups;
  {
    std::shared_lock latch_guard(_non_owning_table->row_group_latch());
    auto const num_row_groups = _non_owning_table->num_row_groups();
    row_groups.reserve(num_row_groups);
    for (std::size_t row_group_idx = 0; row_group_idx < num_row_groups; ++row_group_idx) {
      row_groups.push_back(_non_owning_table->row_group_at(row_group_idx));
    }
  }

  return std::make_unique<oneshot_in_memory_read_task>(std::move(row_groups),
                                                       std::move(column_indexes),
                                                       std::move(data_types),
                                                       std::move(partial_filter));
}

std::optional<arrow::compute::Expression> in_memory_readable_view::transform_partial_filter(
  gqe::expression* partial_filter)
{
  if (!partial_filter) { return std::nullopt; }
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  if (zone_map_filter) {
    GQE_LOG_DEBUG(
      "Using partial filter to prune results\nPartial filter:\n{}\nZone map filter:\n{}",
      expression_json_formatter::to_json(*partial_filter),
      zone_map_filter->ToString());
  }
  return zone_map_filter;
}

in_memory_writeable_view::in_memory_writeable_view(in_memory_table* non_owning_table)
  : writeable_view(), _non_owning_table(non_owning_table)
{
}

in_memory_writeable_view::in_memory_writeable_view(shared_in_memory_table* non_owning_table)
  : writeable_view(), _non_owning_table(non_owning_table)
{
}

std::vector<std::unique_ptr<write_task_base>> in_memory_writeable_view::get_write_tasks(
  std::vector<writeable_view::task_parameters>&& task_parameters,
  context_reference ctx_ref,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  table_statistics_manager* statistics)
{
  assert(!task_parameters.empty() && "Must have at least one write task");
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0 && "Stage ID must be a positive value");
  assert(ctx_ref._task_manager_context != nullptr);

  auto* table =
    std::visit([](auto* concrete_table) -> in_memory_table_interface* { return concrete_table; },
               _non_owning_table);

  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(
    column_names.cbegin(),
    column_names.cend(),
    std::back_inserter(column_indexes),
    [table](std::string const& column_name) { return table->get_column_index(column_name); });

  std::vector<std::unique_ptr<write_task_base>> write_tasks;
  write_tasks.reserve(task_parameters.size());

  auto const memory_kind = table->get_memory_kind();
  auto memory_resource   = ctx_ref._task_manager_context->get_table_memory_resource(memory_kind);

  for (auto& task_parameter : task_parameters) {
    auto row_group_appender = std::visit(
      [](auto* concrete_table) -> in_memory_write_task::row_group_appender_t {
        return concrete_table->get_row_group_appender();
      },
      _non_owning_table);

    write_tasks.push_back(std::make_unique<in_memory_write_task>(ctx_ref,
                                                                 task_parameter.task_id,
                                                                 stage_id,
                                                                 std::move(task_parameter.input),
                                                                 memory_resource,
                                                                 std::move(row_group_appender),
                                                                 memory_kind,
                                                                 column_indexes,
                                                                 data_types,
                                                                 statistics));
  }

  return write_tasks;
}

}  // namespace storage

}  // namespace gqe
