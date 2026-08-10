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

#pragma once

#include <gqe/context_reference.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/executor/write.hpp>
#include <gqe/optimizer/statistics.hpp>
#include <gqe/storage/buffer.hpp>
#include <gqe/storage/column.hpp>
#include <gqe/storage/compression.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/storage/sliced_columns.cuh>
#include <gqe/storage/storage_handle.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/cuda.hpp>

#include <boost/container/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

class query_context;

namespace storage {
class in_memory_readable_view;
class in_memory_writeable_view;
class shared_in_memory_table;
class decompression_batch;
class row_group;
class row_group_interface;

/**
 * @brief Indicate which partitions of an in-memory table remain after partition pruning and which
 * can be skipped.
 *
 * Depending on the column type, the projected columns of an in_memory_read_task compute the same
 * information over the list of partitions returned by the zone map evaluation. For example,
 * contiguous columns consolidate individual partitions into maximally large partitions, whereas
 * compressed sliced columns require individual partition indices. All column classes require the
 * number of rows remaining after partition pruning.
 *
 * This class precomputes these values, so that they are only computed once when accessed by
 * multiple columns.
 */
class pruning_result_t {
 public:
  explicit pruning_result_t(const std::vector<zone_map::partition>& partitions,
                            cudf::size_type partition_size);

  [[nodiscard]] cudf::size_type partition_size() const;

  /// @brief Return all partitions which are not pruned.
  [[nodiscard]] const std::vector<zone_map::partition>& candidate_partitions() const;

  /// @brief Return maximally consolidated partitions.
  [[nodiscard]] const std::vector<zone_map::partition>& consolidated_partitions() const;

  /// @brief Return partition ids
  [[nodiscard]] const std::vector<size_t>& partition_indexes() const;

  /// @brief Return the number of rows after pruning.
  [[nodiscard]] cudf::size_type num_rows() const;

 private:
  std::vector<zone_map::partition> _partitions;
  cudf::size_type _partition_size;
  std::vector<zone_map::partition> _candidate_partitions;
  std::vector<zone_map::partition> _consolidated_partitions;
  std::vector<size_t> _partition_indexes;
  cudf::size_type _num_rows;
};

/// Pair a row group with its partition pruning result
using row_group_with_pruning_result_t = std::pair<const row_group_interface*, pruning_result_t>;

/// Pruning results of all row groups of an in-memory table
using pruning_results_t = std::vector<row_group_with_pruning_result_t>;

/**
 * @brief Physical storage layout for an in-memory column.
 */
enum class column_layout {
  contiguous,
  compressed_sliced,
  string_compressed_sliced,
  large_string_compressed_sliced
};

using contiguous_column_storage        = gqe::column<>;
using shared_contiguous_column_storage = gqe::column<shared_memory_allocator>;

/**
 * @brief Handle to one column's non-polymorphic storage object in Boost shared memory.
 */
class column_storage_handle {
 public:
  column_storage_handle(column_layout layout, shared_memory_handle_t storage_handle);

  column_layout layout;
  shared_memory_handle_t storage_handle;
};

/**
 * @brief Non-polymorphic row-group metadata stored in Boost shared memory.
 *
 * Shared row groups store handles to shared column storage objects and an optional shared zone-map
 * storage handle. Process-local wrappers resolve those handles when reading the row group.
 */
class shared_row_group_storage {
 public:
  using allocator_type   = shared_memory_allocator;
  using allocator_traits = std::allocator_traits<allocator_type>;
  template <class T>
  using rebind_allocator = typename allocator_traits::template rebind_alloc<T>;
  template <class T>
  using vector_type = boost::container::vector<T, rebind_allocator<T>>;

  shared_row_group_storage(cudf::size_type num_rows, allocator_type const& alloc);

  cudf::size_type num_rows;
  vector_type<column_storage_handle> columns;
  std::optional<shared_memory_handle_t> zone_map;
};

/**
 * @brief Non-polymorphic shared in-memory table metadata stored in Boost shared memory.
 *
 * Shared table storage owns stable handles to separately constructed shared row-group storage
 * objects and protects append operations with an interprocess mutex. Process-local table wrappers
 * resolve the handles into `shared_row_group` wrappers.
 *
 * Appending a row-group handle and updating the revision use the same interprocess mutex.
 */
class shared_in_memory_table_storage {
 public:
  using allocator_type   = shared_memory_allocator;
  using allocator_traits = std::allocator_traits<allocator_type>;
  template <class T>
  using rebind_allocator = typename allocator_traits::template rebind_alloc<T>;
  template <class T>
  using vector_type = boost::container::vector<T, rebind_allocator<T>>;

  explicit shared_in_memory_table_storage(allocator_type const& alloc);

  shared_memory_handle_t append(boost::interprocess::managed_shared_memory& segment,
                                shared_row_group_storage&& row_group);

 private:
  friend class shared_in_memory_table;

  mutable boost::interprocess::interprocess_mutex _mutex;
  vector_type<shared_memory_handle_t> _row_group_storage_handles;
  std::uint64_t _storage_revision{};
};

/**
 * @brief Abstract base column of an in-memory table.
 *
 * No guarantees are provided about the memory layout. Concrete implementations are free to choose
 * their own memory layout.
 */
class column_base {
 public:
  column_base()          = default;
  virtual ~column_base() = default;

  column_base(const column_base& other)      = delete;
  column_base& operator=(const column_base&) = delete;

  /**
   * @brief Explicitly release backing storage associated with this column.
   */
  virtual void deallocate(rmm::cuda_stream_view, rmm::device_async_resource_ref) {}

  /**
   * @brief Return the physical layout of the in-memory column.
   */
  [[nodiscard]] virtual column_layout layout() const = 0;

  /**
   * @brief Return the column size.
   */
  [[nodiscard]] virtual int64_t size() const = 0;

  /**
   * @brief Return the null count of the column.
   */
  [[nodiscard]] virtual cudf::size_type null_count() const = 0;

  /**
   * @brief Return whether the column is compressed or not.
   */
  [[nodiscard]] virtual bool is_compressed() const = 0;

  /**
   * @brief Return the compression ratio of the column if it is compressed, otherwise return 1.0.
   */
  [[nodiscard]] virtual double get_compression_ratio() const
  {
    // Guard against division by zero.
    if (get_compressed_size() == 0 || !is_compressed()) { return 1.0; }
    return static_cast<double>(get_uncompressed_size()) /
           static_cast<double>(get_compressed_size());
  }

  /**
   * @brief Return the compressed size of the column in bytes if it is compressed, otherwise return
   * the uncompressed size.
   */
  [[nodiscard]] virtual int64_t get_compressed_size() const = 0;

  /**
   * @brief Return the uncompressed size of the column in bytes.
   */
  [[nodiscard]] virtual int64_t get_uncompressed_size() const = 0;

  /**
   * @brief Return the column compression statistics.
   */
  [[nodiscard]] virtual column_compression_statistics get_compression_stats() const = 0;
};

// clang-format off
/*
 * Contiguous column wrappers:
 *
 *   column_base
 *     `- contiguous_column_interface              (runtime view() API)
 *          |- basic_contiguous_column<local_storage_handle<contiguous_column_storage>>
 *          |    `- contiguous_column              (local facade)
 *          `- basic_contiguous_column<shared_storage_handle<shared_contiguous_column_storage>>
 *               `- shared_contiguous_column       (shared facade)
 *
 */
// clang-format on

/**
 * @brief Process-local interface for contiguous column wrappers.
 */
class contiguous_column_interface : public column_base {
 public:
  [[nodiscard]] column_layout layout() const override { return column_layout::contiguous; }

  /**
   * @brief Return a cuDF-compatible column view.
   */
  [[nodiscard]] virtual cudf::column_view view() const = 0;
};

/**
 * @brief Common implementation for contiguous local and shared column wrappers.
 *
 * The storage handle decides whether storage is owned by value or resolved from shared memory; this
 * class provides the common contiguous-column forwarding behavior on top of that storage access.
 *
 * @tparam StorageHandle Handle type that exposes the concrete contiguous storage.
 */
template <class StorageHandle>
class basic_contiguous_column : public contiguous_column_interface {
 public:
  using storage_type = typename StorageHandle::storage_type;

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  [[nodiscard]] int64_t size() const override;

  /**
   * @copydoc gqe::storage::column_base::null_count()
   */
  [[nodiscard]] cudf::size_type null_count() const override;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   */
  [[nodiscard]] bool is_compressed() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   */
  [[nodiscard]] int64_t get_compressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   */
  [[nodiscard]] int64_t get_uncompressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;

  /**
   * @brief Return a cuDF compatible column view.
   */
  [[nodiscard]] cudf::column_view view() const override;

  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;

 protected:
  // Forward construction args to storage handle to hide details about storage from column wrapper.
  template <class... Args>
  explicit basic_contiguous_column(Args&&... args) : _storage_handle(std::forward<Args>(args)...)
  {
  }

 protected:
  StorageHandle _storage_handle;
};

/**
 * @brief In-memory column with a contiguous memory layout.
 */
class contiguous_column final
  : public basic_contiguous_column<local_storage_handle<contiguous_column_storage>> {
 public:
  using storage_type        = contiguous_column_storage;
  using storage_handle_type = local_storage_handle<storage_type>;
  using base_type           = basic_contiguous_column<storage_handle_type>;

  explicit contiguous_column(cudf::column&& cudf_column);
  contiguous_column(cudf::column&& cudf_column,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr,
                    storage_type::allocator_type const& alloc = storage_type::allocator_type{});
  ~contiguous_column() override = default;

  contiguous_column(const contiguous_column&)            = delete;
  contiguous_column& operator=(const contiguous_column&) = delete;

  /**
   * @brief Return a cuDF compatible mutable column view.
   */
  cudf::mutable_column_view mutable_view();
};

/**
 * @brief Contiguous column in inter-process CPU shared memory.
 */
class shared_contiguous_column final
  : public basic_contiguous_column<shared_storage_handle<shared_contiguous_column_storage>> {
 public:
  using storage_type        = shared_contiguous_column_storage;
  using storage_handle_type = shared_storage_handle<storage_type>;
  using base_type           = basic_contiguous_column<storage_handle_type>;

  shared_contiguous_column(boost::interprocess::managed_shared_memory& segment,
                           typename storage_handle_type::handle_type handle)
    : base_type(segment, handle)
  {
  }

  ~shared_contiguous_column() override = default;

  shared_contiguous_column(const shared_contiguous_column&)            = delete;
  shared_contiguous_column& operator=(const shared_contiguous_column&) = delete;
};

/// Device span containing the bytes for one compressed-sliced partition.
using partition_span = cudf::device_span<uint8_t const>;

/**
 * @brief Non-owning view over one partitioned buffer of a compressed-sliced column.
 *
 * The view can reference either the data buffer or an auxiliary buffer, such as string offsets.
 * It preserves the buffer's compression metadata so callers can copy uncompressed partitions
 * directly or schedule decompression of compressed partitions.
 */
class compressed_sliced_buffer_view {
 public:
  compressed_sliced_buffer_view() = default;

  compressed_sliced_buffer_view(cudf::data_type element_type,
                                std::span<gqe::storage::buffer const> buffers,
                                gqe::compression_format compression_format)
    : _element_type(element_type), _buffers(buffers), _compression_format(compression_format)
  {
  }

  /**
   * @brief Return true if the referenced partition buffer is compressed.
   */
  [[nodiscard]] bool is_compressed() const;

  /**
   * @brief Return the compression format used by the referenced partition buffer.
   */
  [[nodiscard]] gqe::compression_format compression_format() const;

  /**
   * @brief Return the element type represented by the referenced partition buffer.
   */
  [[nodiscard]] cudf::data_type element_type() const;

  /**
   * @brief Return a device span for one partition in the referenced buffer.
   *
   * @param[in] partition_idx Index of the partition to access.
   */
  [[nodiscard]] partition_span get_partition(size_t partition_idx) const;

 private:
  cudf::data_type _element_type{cudf::data_type(cudf::type_id::EMPTY)};
  std::span<gqe::storage::buffer const> _buffers{};
  gqe::compression_format _compression_format{gqe::compression_format::none};
};

/**
 * @brief Compressed and partitioned column storage.
 *
 * This type is allocator-aware and contains no virtual functions so it can be constructed in Boost
 * managed shared memory.
 *
 * @tparam Allocator Allocator used for persistent host-side metadata containers.
 */
template <class Allocator = std::allocator<std::byte>>
class compressed_sliced_column_storage {
 protected:
  struct buffer_storage;

 public:
  using allocator_type = gqe::storage_allocator_t<Allocator>;
  template <class T>
  using vector_type = gqe::storage_vector_t<Allocator, T>;

  using buffer_view = compressed_sliced_buffer_view;

  compressed_sliced_column_storage(cudf::column&& cudf_column,
                                   int partition_size,
                                   memory_kind::type memory_kind,
                                   gqe::compression::configuration compression_config,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr,
                                   allocator_type const& alloc);

  [[nodiscard]] static compressed_sliced_column_storage deserialize_from_disk(
    std::filesystem::path const& file_path,
    std::string const& column_name,
    size_t row_group_index,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    allocator_type const& alloc);

  ~compressed_sliced_column_storage() = default;

  compressed_sliced_column_storage(const compressed_sliced_column_storage&)            = delete;
  compressed_sliced_column_storage& operator=(const compressed_sliced_column_storage&) = delete;
  compressed_sliced_column_storage(compressed_sliced_column_storage&&) noexcept        = default;
  compressed_sliced_column_storage& operator=(compressed_sliced_column_storage&&) noexcept =
    default;

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  [[nodiscard]] int64_t size() const;

  /**
   * @copydoc gqe::storage::column_base::null_count()
   */
  [[nodiscard]] cudf::size_type null_count() const;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   */
  [[nodiscard]] bool is_compressed() const;

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   * Compute the compressed size for the column, considering the data and null mask.
   */
  [[nodiscard]] int64_t get_compressed_size() const;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   * Compute the uncompressed size for the column, considering the data and null mask.
   */
  [[nodiscard]] int64_t get_uncompressed_size() const;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const;

  /**
   * @brief Return a non-owning view of the partitioned data buffer.
   */
  [[nodiscard]] buffer_view data_buffer_view() const
  {
    return buffer_view(_data_buffer.element_type,
                       std::span<gqe::storage::buffer const>{_data_buffer.buffers.data(),
                                                             _data_buffer.buffers.size()},
                       _data_buffer.compression_format);
  }

  /**
   * @brief Return the compression configuration used by this column's partition buffers.
   */
  [[nodiscard]] gqe::compression::configuration const& get_compression_config() const
  {
    return _compression_config;
  }

  /**
   * @brief Explicitly deallocate all backing partition buffers.
   *
   * This is a terminal-use API. Callers must not read from this storage after deallocation.
   */
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

 protected:
  /**
   * @brief Owning storage for the per-partition buffers and their compression metadata.
   */
  struct buffer_storage {
    /**
     * @brief Construct empty storage for an untyped buffer.
     */
    explicit buffer_storage(allocator_type const& alloc)
      : buffers(gqe::storage_rebind_allocator_t<Allocator, gqe::storage::buffer>(alloc))
    {
    }

    /**
     * @brief Construct empty storage for a typed buffer.
     *
     * @param[in] element_type Type of the values represented by this buffer.
     */
    buffer_storage(cudf::data_type element_type, allocator_type const& alloc)
      : element_type(element_type),
        buffers(gqe::storage_rebind_allocator_t<Allocator, gqe::storage::buffer>(alloc))
    {
    }

    /**
     * @brief Deallocate all payload buffers without synchronizing @p stream.
     */
    void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    {
      for (auto& buffer : buffers) {
        buffer.deallocate(stream, mr);
      }
      buffers.clear();
    }

    cudf::data_type element_type{cudf::data_type(cudf::type_id::EMPTY)};
    vector_type<gqe::storage::buffer> buffers;
    gqe::compression_format compression_format{gqe::compression_format::none};
    size_t primary_compressed_size{0};
    size_t secondary_compressed_size{0};
    size_t uncompressed_size{0};
    size_t compressed_size{0};
  };

  size_t _size{0};
  size_t _partition_size{0};
  cudf::data_type _cudf_type;

  vector_type<cudf::size_type> _null_counts;
  gqe::compression::configuration _compression_config;
  buffer_storage _data_buffer;
  buffer_storage _null_mask_buffer;

  // Protected constructor -- this is only called by derived classes (the string sliced column)
  // We fill the base members but don't do compression
  compressed_sliced_column_storage(const cudf::column& cudf_column,
                                   int partition_size,
                                   gqe::compression::configuration compression_config,
                                   allocator_type const& alloc);

  void compress(cudf::column&& cudf_column,
                memory_kind::type memory_kind,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /**
   * @brief Compress one buffer in the column (i.e. data or null mask)
   *
   * @param[in] input The input buffer to compress
   * @param[out] output The compressed buffer
   * @param[in] num_rows The number of rows in the column
   * @param[in] num_partitions The number of partitions in the column
   * @param[in] is_null_mask Whether the column is a null mask
   * @param[in] memory_kind Memory kind used for the compressed output buffers.
   * @param[in] stream The CUDA stream to use
   * @param[in] mr The memory resource to use
   */
  void do_compress(rmm::device_buffer const* input,
                   buffer_storage& output,
                   size_t num_rows,
                   size_t num_partitions,
                   bool is_null_mask,
                   memory_kind::type memory_kind,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr);

 private:
  /**
   * @brief Reconstruct storage from already-loaded compressed partition buffers.
   */
  compressed_sliced_column_storage(size_t size,
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
                                   allocator_type const& alloc);
};

using shared_compressed_sliced_column_storage =
  compressed_sliced_column_storage<shared_memory_allocator>;

// clang-format off
/*
 * Compressed-sliced column wrappers:
 *
 *   column_base
 *     `- compressed_sliced_column_interface                  (runtime compressed-buffer API)
 *          |- basic_compressed_sliced_column<local_storage_handle<compressed_sliced_column_storage<>>>
 *          |    `- compressed_sliced_column                  (local facade)
 *          |- basic_compressed_sliced_column<shared_storage_handle<shared_compressed_sliced_column_storage>>
 *          |    `- shared_compressed_sliced_column           (shared facade)
 *          `- string_compressed_sliced_column_interface      (runtime string-buffer API)
 *               |- basic_string_compressed_sliced_column<large_string_mode, local_storage_handle<...>>
 *               |    `- string_compressed_sliced_column<large_string_mode>
 *               `- basic_string_compressed_sliced_column<large_string_mode, shared_storage_handle<...>>
 *                    `- shared_string_compressed_sliced_column<large_string_mode>
 *
 */
// clang-format on

/**
 * @brief Process-local interface for compressed sliced column wrappers.
 */
class compressed_sliced_column_interface : public column_base {
 public:
  using buffer_view = compressed_sliced_buffer_view;

  [[nodiscard]] column_layout layout() const override { return column_layout::compressed_sliced; }

  [[nodiscard]] virtual buffer_view data_buffer_view() const                                  = 0;
  [[nodiscard]] virtual gqe::compression::configuration const& get_compression_config() const = 0;

  void serialize_to_disk(std::string const& column_name,
                         std::filesystem::path const& file_path,
                         size_t row_group_index,
                         rmm::cuda_stream_view stream) const;
};

/**
 * @brief Common implementation for compressed-sliced local and shared column wrappers.
 *
 * The storage handle abstracts whether the compressed partition buffers live in local storage or in
 * Boost managed shared memory. The wrapper behavior remains identical for both cases.
 *
 * @tparam StorageHandle Handle type that exposes the concrete compressed-sliced storage.
 */
template <class StorageHandle>
class basic_compressed_sliced_column : public compressed_sliced_column_interface {
 public:
  using storage_type = typename StorageHandle::storage_type;
  using buffer_view  = typename storage_type::buffer_view;

  [[nodiscard]] int64_t size() const override;
  [[nodiscard]] cudf::size_type null_count() const override;
  [[nodiscard]] bool is_compressed() const override;
  [[nodiscard]] int64_t get_compressed_size() const override;
  [[nodiscard]] int64_t get_uncompressed_size() const override;
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;
  [[nodiscard]] buffer_view data_buffer_view() const override;
  [[nodiscard]] gqe::compression::configuration const& get_compression_config() const override;
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;

 protected:
  // Forward construction args to storage handle to hide details about storage from column wrapper.
  template <class... Args>
  explicit basic_compressed_sliced_column(Args&&... args)
    : _storage_handle(std::forward<Args>(args)...)
  {
  }

 protected:
  StorageHandle _storage_handle;
};

/**
 * @brief Local in-memory wrapper for compressed sliced column storage.
 */
class compressed_sliced_column final : public basic_compressed_sliced_column<
                                         local_storage_handle<compressed_sliced_column_storage<>>> {
 public:
  using storage_type        = compressed_sliced_column_storage<>;
  using storage_handle_type = local_storage_handle<storage_type>;
  using base_type           = basic_compressed_sliced_column<storage_handle_type>;
  using allocator_type      = typename storage_type::allocator_type;
  using buffer_view         = typename storage_type::buffer_view;

  compressed_sliced_column(cudf::column&& cudf_column,
                           int partition_size,
                           memory_kind::type memory_kind,
                           gqe::compression::configuration compression_config,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr,
                           allocator_type const& alloc = allocator_type{})
    : base_type(std::move(cudf_column),
                partition_size,
                memory_kind,
                std::move(compression_config),
                stream,
                mr,
                alloc)
  {
  }

  [[nodiscard]] static std::unique_ptr<compressed_sliced_column> deserialize_from_disk(
    std::filesystem::path const& file_path,
    std::string const& column_name,
    size_t row_group_index,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

 private:
  explicit compressed_sliced_column(storage_type&& storage) : base_type(std::move(storage)) {}
};

/**
 * @brief Process-local wrapper over compressed sliced storage in Boost managed shared memory.
 */
class shared_compressed_sliced_column final
  : public basic_compressed_sliced_column<
      shared_storage_handle<shared_compressed_sliced_column_storage>> {
 public:
  using storage_type        = shared_compressed_sliced_column_storage;
  using storage_handle_type = shared_storage_handle<storage_type>;
  using base_type           = basic_compressed_sliced_column<storage_handle_type>;

  shared_compressed_sliced_column(boost::interprocess::managed_shared_memory& segment,
                                  typename storage_handle_type::handle_type handle)
    : base_type(segment, handle)
  {
  }
};

/**
 * @brief Collects partition decompression requests and enqueues them sequentially.
 *
 * This is paired with @ref gqe::utility::copy_batch: compressed partition bytes are first copied
 * into staging buffers, then the decompression batch expands those staged bytes into final output
 * buffers.
 */
class decompression_batch {
 public:
  /**
   * @brief Add a decompression request to the batch.
   *
   * Empty partition lists are ignored.
   *
   * @param[in] host_source_buffer Source partition buffer view.
   * @param[in] config Compression configuration owned by the source column.
   * @param[in] partition_indexes Partition indices represented by @p device_staging_buffer.
   * @param[in] device_staging_buffer Temporary buffer containing compressed partition bytes.
   * @param[out] output_ptr Destination pointer for decompressed bytes.
   */
  void add(compressed_sliced_buffer_view host_source_buffer,
           gqe::compression::configuration config,
           std::vector<size_t> partition_indexes,
           rmm::device_buffer* device_staging_buffer,
           std::byte* output_ptr);

  /**
   * @brief Reserve storage for at least @p num_requests decompression requests.
   */
  void reserve(size_t num_requests);

  /**
   * @brief Return true if the batch contains no decompression requests.
   */
  [[nodiscard]] bool empty() const;

  /**
   * @brief Return the number of decompression requests in the batch.
   */
  [[nodiscard]] size_t size() const;

  /**
   * @brief Execute all queued decompressions on a CUDA stream.
   *
   * This synchronizes @p stream before returning so request-local nvCOMP metadata remains alive
   * until decompression completes.
   *
   * @param[in] stream CUDA stream used for decompression.
   */
  void execute(rmm::cuda_stream_view stream) const;

 private:
  /**
   * @brief Non-owning metadata for one partition decompression operation.
   */
  struct decompression_request {
    compressed_sliced_buffer_view host_source_buffer;
    gqe::compression::configuration config;
    std::vector<size_t> partition_indexes;
    rmm::device_buffer* device_staging_buffer;
    std::byte* output_ptr;
  };

  std::vector<decompression_request> _requests;
};

/**
 * @brief Compressed sliced string column storage.
 *
 * This is a compressed (and sliced) in-memory column that contains a string column.
 * It is used to store string columns in a compressed and sliced format.
 *
 * This is different because the char (data) array / offset need to be compressed separately
 * Under decompression the offsets also need to be adjusted to point correctly to the reduced char
 * array
 *
 * Partition size must be a multiple of 32
 *
 * @tparam large_string_mode Whether the string column is large (i.e. int64_t offsets)
 * @tparam Allocator Allocator used for persistent host-side metadata containers.
 */
template <bool large_string_mode, class Allocator = std::allocator<std::byte>>
class string_compressed_sliced_column_storage : public compressed_sliced_column_storage<Allocator> {
  using base_type = compressed_sliced_column_storage<Allocator>;

 public:
  using allocator_type = typename base_type::allocator_type;
  template <class T>
  using vector_type    = typename base_type::template vector_type<T>;
  using buffer_storage = typename base_type::buffer_storage;
  using buffer_view    = typename base_type::buffer_view;

  string_compressed_sliced_column_storage(cudf::column&& cudf_column,
                                          int partition_size,
                                          memory_kind::type memory_kind,
                                          gqe::compression::configuration compression_config,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr,
                                          allocator_type const& alloc);

  /// @copydoc string_compressed_sliced_column_interface::is_large_string
  [[nodiscard]] bool is_large_string() const;

  /// Represent the size of the character offsets.
  using offsets_type = std::conditional_t<large_string_mode, int64_t, int32_t>;
  static constexpr cudf::data_type offset_element_type = large_string_mode
                                                           ? cudf::data_type(cudf::type_id::INT64)
                                                           : cudf::data_type(cudf::type_id::INT32);

  /**
   * @brief Compress the string column's character and offset buffers independently.
   *
   * @param[in] cudf_column String column to compress.
   * @param[in] memory_kind Memory kind used for compressed buffers.
   * @param[in] stream CUDA stream used for compression.
   * @param[in] mr Memory resource used for temporary and output allocations.
   */
  void compress(cudf::column&& cudf_column,
                memory_kind::type memory_kind,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /**
   * @brief Return a non-owning view of the partitioned character data buffer.
   */
  [[nodiscard]] buffer_view char_buffer_view() const { return this->data_buffer_view(); }

  /**
   * @brief Return the decompressed character byte count for one partition.
   *
   * @param[in] partition_idx Partition index to query.
   */
  [[nodiscard]] size_t get_char_partition_output_size(size_t partition_idx) const;

  /**
   * @brief Return a non-owning view of the partitioned string offset buffer.
   */
  [[nodiscard]] buffer_view offset_buffer_view() const
  {
    return buffer_view(_offset_buffer.element_type,
                       std::span<gqe::storage::buffer const>{_offset_buffer.buffers.data(),
                                                             _offset_buffer.buffers.size()},
                       _offset_buffer.compression_format);
  }

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   * Compute the compressed size for the column, considering the data, offsets, and null mask.
   */
  [[nodiscard]] int64_t get_compressed_size() const;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   * Compute the uncompressed size for the column, considering the data, offsets, and null mask.
   */
  [[nodiscard]] int64_t get_uncompressed_size() const;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   * Return true if either the data or offsets are compressed.
   */
  [[nodiscard]] bool is_compressed() const;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const;

  /**
   * @brief Explicitly deallocate all backing partition buffers.
   *
   * This is a terminal-use API. Callers must not read from this storage after deallocation.
   */
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

 private:
  buffer_storage _offset_buffer;
  vector_type<cudf::size_type> _partition_char_array_sizes;
};

template <bool large_string_mode>
using shared_string_compressed_sliced_column_storage =
  string_compressed_sliced_column_storage<large_string_mode, shared_memory_allocator>;

/**
 * @brief Base class for process-local compressed sliced string column wrappers.
 */
class string_compressed_sliced_column_interface : public compressed_sliced_column_interface {
 public:
  /**
   * @brief Determine if the offset type is 64-bit or 32-bit.
   * @return True, if the offset type is 64-bit; false, otherwise.
   */
  [[nodiscard]] virtual bool is_large_string() const = 0;

  [[nodiscard]] virtual buffer_view char_buffer_view() const                              = 0;
  [[nodiscard]] virtual buffer_view offset_buffer_view() const                            = 0;
  [[nodiscard]] virtual size_t get_char_partition_output_size(size_t partition_idx) const = 0;
};

/**
 * @brief Common implementation for string compressed-sliced local and shared column wrappers.
 *
 * This folds the common string-specific forwarding behavior while preserving separate public
 * façade types for local and shared string compressed-sliced columns.
 *
 * @tparam large_string_mode Whether the wrapped storage uses 64-bit string offsets.
 * @tparam StorageHandle Handle type that exposes the concrete string compressed-sliced storage.
 */
template <bool large_string_mode, class StorageHandle>
class basic_string_compressed_sliced_column : public string_compressed_sliced_column_interface {
 public:
  using storage_type = typename StorageHandle::storage_type;
  using buffer_view  = typename storage_type::buffer_view;
  using offsets_type = typename storage_type::offsets_type;

  [[nodiscard]] column_layout layout() const override;
  [[nodiscard]] int64_t size() const override;
  [[nodiscard]] cudf::size_type null_count() const override;
  [[nodiscard]] bool is_compressed() const override;
  [[nodiscard]] int64_t get_compressed_size() const override;
  [[nodiscard]] int64_t get_uncompressed_size() const override;
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;
  [[nodiscard]] bool is_large_string() const override;
  [[nodiscard]] buffer_view data_buffer_view() const override;
  [[nodiscard]] buffer_view char_buffer_view() const override;
  [[nodiscard]] buffer_view offset_buffer_view() const override;
  [[nodiscard]] size_t get_char_partition_output_size(size_t partition_idx) const override;
  [[nodiscard]] gqe::compression::configuration const& get_compression_config() const override;
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;

 protected:
  // Forward construction args to storage handle to hide details about storage from column wrapper.
  template <class... Args>
  explicit basic_string_compressed_sliced_column(Args&&... args)
    : _storage_handle(std::forward<Args>(args)...)
  {
  }

 protected:
  StorageHandle _storage_handle;
};

/**
 * @brief Local in-memory wrapper for compressed sliced string column storage.
 */
template <bool large_string_mode>
class string_compressed_sliced_column final
  : public basic_string_compressed_sliced_column<
      large_string_mode,
      local_storage_handle<string_compressed_sliced_column_storage<large_string_mode>>> {
 public:
  using storage_type        = string_compressed_sliced_column_storage<large_string_mode>;
  using storage_handle_type = local_storage_handle<storage_type>;
  using base_type = basic_string_compressed_sliced_column<large_string_mode, storage_handle_type>;
  using allocator_type = typename storage_type::allocator_type;
  using buffer_view    = typename storage_type::buffer_view;
  using offsets_type   = typename storage_type::offsets_type;

  string_compressed_sliced_column(cudf::column&& cudf_column,
                                  int partition_size,
                                  memory_kind::type memory_kind,
                                  gqe::compression::configuration compression_config,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr,
                                  allocator_type const& alloc = allocator_type{})
    : base_type(std::move(cudf_column),
                partition_size,
                memory_kind,
                std::move(compression_config),
                stream,
                mr,
                alloc)
  {
  }
};

/**
 * @brief Process-local wrapper over compressed sliced string storage in shared memory.
 */
template <bool large_string_mode>
class shared_string_compressed_sliced_column final
  : public basic_string_compressed_sliced_column<
      large_string_mode,
      shared_storage_handle<shared_string_compressed_sliced_column_storage<large_string_mode>>> {
 public:
  using storage_type        = shared_string_compressed_sliced_column_storage<large_string_mode>;
  using storage_handle_type = shared_storage_handle<storage_type>;
  using base_type = basic_string_compressed_sliced_column<large_string_mode, storage_handle_type>;

  shared_string_compressed_sliced_column(boost::interprocess::managed_shared_memory& segment,
                                         typename storage_handle_type::handle_type handle)
    : base_type(segment, handle)
  {
  }
};

extern template class compressed_sliced_column_storage<std::allocator<std::byte>>;
extern template class compressed_sliced_column_storage<shared_memory_allocator>;
extern template class string_compressed_sliced_column_storage<false, std::allocator<std::byte>>;
extern template class string_compressed_sliced_column_storage<true, std::allocator<std::byte>>;
extern template class string_compressed_sliced_column_storage<false, shared_memory_allocator>;
extern template class string_compressed_sliced_column_storage<true, shared_memory_allocator>;

/**
 * @brief Common row-group behavior consumed by in-memory read tasks.
 *
 * Implementations may own local column wrappers directly or resolve shared-memory storage handles
 * through process-local wrappers. Callers use the interface uniformly when building read results.
 */
class row_group_interface {
 public:
  row_group_interface()          = default;
  virtual ~row_group_interface() = default;

  row_group_interface(row_group_interface const&)            = delete;
  row_group_interface& operator=(row_group_interface const&) = delete;
  row_group_interface(row_group_interface&&)                 = default;
  row_group_interface& operator=(row_group_interface&&)      = default;

  /**
   * @brief Return the number of rows in the row group.
   */
  [[nodiscard]] virtual int64_t size() const = 0;

  [[nodiscard]] virtual int64_t num_columns() const = 0;

  /**
   * @brief Lookup a column by index.
   */
  [[nodiscard]] virtual column_base& get_column(cudf::size_type column_index) const = 0;

  [[nodiscard]] virtual gqe::zone_map* zone_map() const = 0;

  virtual void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) = 0;
};

/**
 * @brief Process-local row group backed by local column and zone-map wrappers.
 */
class row_group : public row_group_interface {
 public:
  row_group()                  = default;
  row_group(row_group&& other) = default;

  row_group(cudf::size_type num_rows,
            std::vector<std::unique_ptr<column_base>>&& columns,
            std::unique_ptr<gqe::zone_map> zone_map);
  explicit row_group(std::vector<std::unique_ptr<column_base>>&& columns);
  explicit row_group(std::vector<std::unique_ptr<column_base>>&& columns,
                     std::unique_ptr<gqe::zone_map> zone_map);

  row_group(const row_group& other)      = delete;
  row_group& operator=(const row_group&) = delete;

  ~row_group() override = default;

  /**
   * @brief Return the number of rows in the row group.
   */
  [[nodiscard]] int64_t size() const override;

  [[nodiscard]] int64_t num_columns() const override;

  /**
   * @brief Lookup a column by index.
   */
  [[nodiscard]] column_base& get_column(cudf::size_type column_index) const override;

  [[nodiscard]] gqe::zone_map* zone_map() const override;

  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;

 private:
  cudf::size_type _num_rows{0};
  std::vector<std::unique_ptr<column_base>> _columns;
  std::unique_ptr<gqe::zone_map> _zone_map;
};

/**
 * @brief Process-local row-group wrapper over metadata stored in Boost shared memory.
 *
 * The wrapper holds a segment pointer and row-group storage handle, then creates local
 * shared-column and shared-zone-map wrappers from the handles in shared row-group storage.
 * Destroying this object destroys only those local wrappers, not the shared storage they reference.
 */
class shared_row_group : public row_group_interface {
 public:
  shared_row_group(boost::interprocess::managed_shared_memory& segment,
                   shared_memory_handle_t shared_row_group_storage_handle);
  ~shared_row_group() override = default;

  shared_row_group(shared_row_group&& other) noexcept            = default;
  shared_row_group& operator=(shared_row_group&& other) noexcept = default;
  shared_row_group(shared_row_group const&)                      = delete;
  shared_row_group& operator=(shared_row_group const&)           = delete;

  [[nodiscard]] int64_t size() const override;
  [[nodiscard]] int64_t num_columns() const override;
  [[nodiscard]] column_base& get_column(cudf::size_type column_index) const override;
  [[nodiscard]] gqe::zone_map* zone_map() const override;
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;

 private:
  [[nodiscard]] shared_row_group_storage const& storage() const;

  boost::interprocess::managed_shared_memory* _segment{};
  shared_memory_handle_t _shared_row_group_storage_handle{};
  std::vector<std::unique_ptr<column_base>> _columns;
  std::unique_ptr<gqe::zone_map> _zone_map;
};

/**
 * @brief Common table interface implemented by local and shared in-memory table wrappers.
 *
 * Read views depend on this interface so they can split row groups and construct read tasks without
 * knowing whether the underlying table is process-local or backed by Boost shared memory.
 */
class in_memory_table_interface : public table {
 public:
  in_memory_table_interface()           = default;
  ~in_memory_table_interface() override = default;

  in_memory_table_interface(in_memory_table_interface const&)            = delete;
  in_memory_table_interface& operator=(in_memory_table_interface const&) = delete;

  [[nodiscard]] virtual cudf::size_type get_column_index(std::string const& column_name) const = 0;
  [[nodiscard]] virtual memory_kind::type get_memory_kind() const                              = 0;
  [[nodiscard]] virtual std::size_t num_row_groups() const                                     = 0;
  [[nodiscard]] virtual row_group_interface const* row_group_at(std::size_t index) const       = 0;
  [[nodiscard]] virtual std::shared_mutex& row_group_latch() const                             = 0;
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override    = 0;
};

/**
 * @brief A table that stores data in memory.
 *
 * The data in this table are represented as a memory location. This includes
 * GPU memory, CPU memory, and memory-mapped storage.
 *
 * == Memory Layout ==
 *
 * The table consists of multiple row groups, each of which contains a column
 * per table attribute. The table does not define the column's memory layout,
 * this is defined by the concrete column type.
 *
 * Row groups are stored in an extensible data structure that maintains reference
 * validity during append operations, e.g., a `std::deque`.
 *
 * == Memory Kinds ==
 *
 * The table provides a memory resource for new memory allocations. The memory
 * kind determines the type of memory resource used. Examples include a CUDA
 * device memory allocator and a NUMA-aware host memory allocator.
 *
 * Some memory kinds are directly accessible by GPU kernels, others are not.
 * Thus, the executor may need to, e.g., copy data to device memory before the
 * kernel reads the data. The data access method must be determined at runtime
 * depending on the memory kind and hardware coherency support.
 *
 * == Thread Safety ==
 *
 * The table guarantees atomicity of appends, and reference validity during
 * appends. The table ensures this by a RW latch on the row group data
 * structure, that allows either multiple readers or a single writer.
 */
class in_memory_table : public in_memory_table_interface {
 public:
  friend in_memory_readable_view;
  friend in_memory_writeable_view;

  /**
   * @brief A functor to append row groups.
   *
   * The row group appender is a separation of concerns: it provides a
   * thread-safe append operation, but not full access to the table.
   *
   * == Thread Safety ==
   *
   * The appender acquires exclusive access to the row group data structure for
   * the duration of the append operation.
   */
  class row_group_appender {
    friend class in_memory_table;

   public:
    /**
     * @brief Append a row group to the table.
     */
    void operator()(row_group&& new_row_group);

    /**
     * @brief Append multiple row groups to the table.
     *
     * This method avoids acquiring the exclusive access multiple times.
     */
    void operator()(std::vector<row_group>&& new_row_groups);

   private:
    row_group_appender(std::deque<row_group>* non_owning_row_groups,
                       std::shared_mutex* non_owning_row_group_latch);

    std::deque<row_group>* _non_owning_row_groups;
    std::shared_mutex* _non_owning_row_group_latch;
  };

  in_memory_table()                             = delete;
  in_memory_table(const in_memory_table& other) = delete;

  /**
   * @brief Create an in-memory table (empty row groups).
   *
   * To load serialized row-group snapshots from disk, call @ref
   * gqe::catalog::deserialize_table after registering the table.
   *
   * @param[in] memory_kind The memory kind to allocate.
   * @param[in] column_names The name per column contained in the table.
   * @param[in] column_types The data type of each column.
   * @param[in] ctx Non-owning pointer to task manager context for accessing centralized
   *                memory resources. Required for boost_shared, numa, and numa_pinned memory kinds.
   */
  in_memory_table(memory_kind::type memory_kind,
                  std::vector<std::string> const& column_names,
                  std::vector<cudf::data_type> const& column_types,
                  task_manager_context* ctx);
  ~in_memory_table() override;

  /**
   * @brief Load serialized row groups from disk under @p table_serialized_data_root (`zmps-*`).
   *
   * For each `rg-{i}/`, zone maps under `zone_maps/` are loaded before column snapshots under
   * `{compression}/{chunk}/`. Column order and names come from @ref _column_names. Replaces any
   * existing row groups.
   *
   * @throw std::runtime_error if loading any row group or column fails.
   */
  void deserialize_table_from_disk(std::string const& table_serialized_data_root,
                                   std::string const& table_name,
                                   rmm::cuda_stream_view stream);

  /**
   * @brief Serialize all row groups to disk under @p table_serialized_data_root (`zmps-*`).
   *
   * For each `rg-{i}/`, writes `zone_maps/` then column `.bin` / `.json` under
   * `{compression}/{chunk}/`. Column order and names come from @ref _column_names. Call after write
   * work on @p stream is complete. Uses a shared latch on the row-group container for the duration
   * of the pass.
   *
   * @throw std::runtime_error if any row group fails to serialize.
   */
  void serialize_table_to_disk(std::string const& table_serialized_data_root,
                               std::string const& table_name,
                               rmm::cuda_stream_view stream);

  /**
   * @copydoc gqe::storage::table::is_readable()
   */
  [[nodiscard]] bool is_readable() const override;

  /**
   * @copydoc gqe::storage::table::is_writeable()
   */
  [[nodiscard]] bool is_writeable() const override;

  /**
   * @copydoc gqe::storage::table::max_concurrent_readers()
   */
  [[nodiscard]] int32_t max_concurrent_readers() const override;

  /**
   * @copydoc gqe::storage::table::max_concurrent_writers()
   */
  [[nodiscard]] int32_t max_concurrent_writers() const override;

  /**
   * @copydoc gqe::storage::table::readable_view()
   */
  std::unique_ptr<storage::readable_view> readable_view() override;

  /**
   * @copydoc gqe::storage::table::writeable_view()
   */
  std::unique_ptr<storage::writeable_view> writeable_view() override;

  /**
   * @copydoc gqe::storage::table::deallocate()
   */
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;

  /**
   * @brief Return the numeric index of a column referenced by its column name.
   *
   * @throw std::out_of_range exception if the column name does not exist.
   */
  [[nodiscard]] cudf::size_type get_column_index(std::string const& column_name) const override;
  [[nodiscard]] memory_kind::type get_memory_kind() const override;
  [[nodiscard]] std::size_t num_row_groups() const override;
  [[nodiscard]] row_group_interface const* row_group_at(std::size_t index) const override;
  [[nodiscard]] std::shared_mutex& row_group_latch() const override;

  row_group_appender get_row_group_appender();

 private:
  task_manager_context* _task_manager_context; /**< Non-owning pointer to the task manager context.
                                                    Used to access memory resources. */
  memory_kind::type _memory_kind;              /**< Memory kind of this table. */
  std::unordered_map<std::string, cudf::size_type> _column_name_to_index;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _column_types;
  std::deque<row_group>
    _row_groups; /** Non-owning references to row_group instances require a container that does not
                  * invalidate references to elements when resizing. `std::deque` guarantees
                  * reference validity when resizing, `std::vector` does not. */
  mutable std::shared_mutex _row_group_latch; /**< An exclusive latch must be held while resizing
                                               * the `std::deque<row_group>`. A shared latch must be
                                               * held for read-only container operations. */
};

/**
 * @brief In-memory table wrapper backed by Boost shared-memory metadata.
 *
 * This wrapper does not create or open the shared-memory segment. The catalog or remote table cache
 * passes an already-open segment and a handle to `shared_in_memory_table_storage`. Construction
 * resolves the currently published row-group storage handles into a process-local snapshot of
 * `shared_row_group` wrappers. The facade is not updated after construction.
 *
 * TODO: Separate shared-storage ownership from immutable process-local read snapshots.
 *
 * Only the catalog owns the shared storage. Remote wrappers do not.
 */
class shared_in_memory_table : public in_memory_table_interface {
 public:
  friend in_memory_readable_view;
  friend in_memory_writeable_view;

  /**
   * @brief Appends row-group storage metadata to a shared in-memory table.
   *
   * The appender holds non-owning access to the shared-memory segment and table storage handle.
   * Appends are synchronized by `shared_in_memory_table_storage`.
   */
  class row_group_appender {
    friend class shared_in_memory_table;

   public:
    void operator()(shared_row_group_storage&& row_group);
    [[nodiscard]] boost::interprocess::managed_shared_memory& segment() const;

   private:
    row_group_appender(boost::interprocess::managed_shared_memory* non_owning_segment,
                       shared_memory_handle_t table_handle);

    boost::interprocess::managed_shared_memory* _non_owning_segment{};
    shared_memory_handle_t _table_handle{};
  };

  shared_in_memory_table(boost::interprocess::managed_shared_memory* non_owning_segment,
                         std::vector<std::string> const& column_names,
                         std::vector<cudf::data_type> const& column_types,
                         shared_memory_handle_t table_handle);
  ~shared_in_memory_table() override = default;

  [[nodiscard]] bool is_readable() const override;
  [[nodiscard]] bool is_writeable() const override;
  [[nodiscard]] std::uint64_t storage_revision() const override;
  [[nodiscard]] int32_t max_concurrent_readers() const override;
  [[nodiscard]] int32_t max_concurrent_writers() const override;
  std::unique_ptr<storage::readable_view> readable_view() override;
  std::unique_ptr<storage::writeable_view> writeable_view() override;

  [[nodiscard]] cudf::size_type get_column_index(std::string const& column_name) const override;
  [[nodiscard]] memory_kind::type get_memory_kind() const override;
  [[nodiscard]] std::size_t num_row_groups() const override;
  [[nodiscard]] row_group_interface const* row_group_at(std::size_t index) const override;
  [[nodiscard]] std::shared_mutex& row_group_latch() const override;

  row_group_appender get_row_group_appender();
  [[nodiscard]] shared_memory_handle_t shared_table_handle() const;
  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;

 private:
  /**
   * @brief Atomically replace all published row groups with @p new_row_group.
   *
   * Publishes @p new_row_group as the table's sole row group, advances `storage_revision` exactly
   * once, and returns ownership of the detached row-group handles to the caller. All potentially
   * throwing preparation happens before the handle swap; an exception leaves shared metadata
   * unchanged. Once this function returns, shared metadata owns @p new_row_group and the caller
   * must not release its column handles.
   *
   * Reclaiming the detached storage is safe only when no reader holds a live wrapper over it. For
   * the statistics tables, refresh and query execution are serialized by the node manager. Do not
   * use this operation on tables that may be read concurrently with replacement.
   *
   * @param[in] new_row_group The single row group to publish (built in this table's segment).
   * @return Handles detached by publication. The caller owns their cleanup.
   */
  [[nodiscard]] std::vector<shared_memory_handle_t> replace_row_groups(
    shared_row_group_storage&& new_row_group);

  /**
   * @brief Reclaim row groups already detached from shared table metadata.
   *
   * Drops this facade's stale local wrappers, then releases the detached row groups. Cleanup
   * failures are logged and contained because ownership of any replacement has already transferred
   * to shared metadata. If local wrappers cannot be dropped, cleanup is skipped to avoid
   * invalidating them.
   */
  void cleanup_detached_row_groups(std::vector<shared_memory_handle_t> const& handles,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr) noexcept;

 public:
  /**
   * @brief Remove all published row groups, leaving the table empty, advancing `storage_revision`
   * once. Frees the previously published row groups from the segment.
   *
   * Same concurrency caveat as @ref replace_row_groups. Used when a statistics refresh has no rows
   * to publish (e.g. no source tables).
   *
   * @param[in] stream CUDA stream used to free the previous row groups' device resources.
   * @param[in] mr Device memory resource used to free the previous row groups.
   */
  void clear_row_groups(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Replace the table's entire contents with @p input_table as a single contiguous,
   * uncompressed row group, or empty the table when @p input_table has no rows.
   *
   * Builds one shared row group with every column stored contiguously (no compression, no zone
   * map) and publishes it via @ref replace_row_groups, advancing `storage_revision` once. Intended
   * for the small, fully-regenerated statistics system tables.
   *
   * Constructs shared-memory storage, so it must be called only on the writing rank (rank 0); see
   * the concurrency caveat on @ref replace_row_groups.
   *
   * @param[in] input_table Rows to publish; its column count must match the table schema.
   * @param[in] memory_resource Device memory resource used to stage the copies into shared memory.
   * @param[in] stream CUDA stream for the copies (synchronized before publishing).
   */
  void replace_contents(cudf::table_view input_table,
                        rmm::device_async_resource_ref memory_resource,
                        rmm::cuda_stream_view stream);

 private:
  [[nodiscard]] shared_in_memory_table_storage const& storage() const;
  [[nodiscard]] shared_in_memory_table_storage& storage();

  memory_kind::type _memory_kind{memory_kind::boost_shared{}};
  std::unordered_map<std::string, cudf::size_type> _column_name_to_index;
  std::vector<cudf::data_type> _column_types;
  std::deque<shared_row_group> _row_groups;
  mutable std::shared_mutex
    _row_group_latch;  ///< Protects this process-local facade's row-group wrappers.
  boost::interprocess::managed_shared_memory* _shared_segment{};
  shared_memory_handle_t _shared_table_handle{};
};

/**
 * @brief Builds one output cuDF column for an in-memory read task.
 *
 * This is an abstract interface. Concrete subclasses implement layout-specific materialization by
 * overriding make_cudf_column() and submit_materialization_requests().
 *
 * Implementations queue any required partition copies and decompression requests, then construct a
 * cuDF column from the selected partitions.
 */
class output_column_builder {
 public:
  /**
   * @brief Construct a builder for one projected output column.
   *
   * @param[in] cudf_type Expected output cuDF data type.
   * @param[in] pruning_results Row groups and partitions selected by predicate pruning.
   * @param[in] column_idx Index of the source column within each row group.
   */
  output_column_builder(cudf::data_type cudf_type,
                        std::shared_ptr<pruning_results_t> pruning_results,
                        cudf::size_type column_idx);

  /**
   * @brief Destroy an output column builder.
   */
  virtual ~output_column_builder() = default;

  /**
   * @brief Queue materialization work required before constructing the output column.
   *
   * Builders for partitioned layouts append copy and decompression requests here. The default
   * implementation does nothing for builders that materialize data through another path.
   *
   * @param[in,out] c_batch Batched copy requests to append to.
   * @param[in,out] d_batch Batched decompression requests to append to.
   * @param[in] stream CUDA stream used for any allocations made while preparing materialization.
   * @param[in] mr Memory resource used for device allocations.
   */
  virtual void submit_materialization_requests(utility::copy_batch& c_batch,
                                               decompression_batch& d_batch,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

  /**
   * @brief Construct the final cuDF column for this output column.
   *
   * @param[in] concatenation_stream CUDA stream used for cuDF column construction or concatenation.
   * @param[in] mr Memory resource used for device allocations.
   * @return The materialized output column.
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream, rmm::device_async_resource_ref mr) = 0;

 protected:
  const cudf::data_type _cudf_type;
  std::shared_ptr<pruning_results_t> _pruning_results;
  const cudf::size_type _column_idx;
  const size_t _num_rows;  //< Number of rows in the output column
};

/**
 * @brief Output column builder that constructs results with @ref cudf::concatenate.
 *
 * This fallback path handles layouts without a specialized partition-copy builder.
 */
class concatenating_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream, rmm::device_async_resource_ref mr) override;
};

/**
 * @brief Output column builder for fixed-width contiguous columns.
 *
 * Selected partitions are copied directly into one contiguous output data buffer.
 *
 * @tparam T Source column type exposing a cuDF-compatible @c view() method.
 */
template <typename T>
class contiguous_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  /**
   * @copydoc gqe::storage::output_column_builder::submit_materialization_requests()
   */
  void submit_materialization_requests(utility::copy_batch& c_batch,
                                       decompression_batch& d_batch,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) override;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream, rmm::device_async_resource_ref mr) override;

 private:
  /**
   * @brief Allocate the output data buffer.
   *
   * @param[in] stream CUDA stream used for allocation.
   * @return The output data buffer.
   */
  [[nodiscard]] rmm::device_buffer allocate_output_buffer(rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr);

  rmm::device_buffer _output_buffer;
};

/**
 * @brief Output column builder for fixed-width compressed-sliced columns.
 *
 * Selected uncompressed partitions are copied directly. Selected compressed partitions are copied
 * into staging buffers and decompressed into the final output buffer.
 */
class compressed_sliced_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  /**
   * @copydoc gqe::storage::output_column_builder::submit_materialization_requests()
   */
  void submit_materialization_requests(utility::copy_batch& c_batch,
                                       decompression_batch& d_batch,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) override;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream, rmm::device_async_resource_ref mr) override;

 private:
  /**
   * @brief Allocate the output data buffer.
   *
   * @param[in] stream CUDA stream used for allocation.
   * @return The output data buffer.
   */
  [[nodiscard]] rmm::device_buffer allocate_output_buffer(rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr);

  rmm::device_buffer _output_buffer;
  std::deque<rmm::device_buffer> _staging_buffers;
};

/**
 * @brief Output column builder for compressed-sliced string columns.
 *
 * Character data and string offsets are materialized through separate partitioned buffers. The
 * builder then adjusts offsets so that the selected character partitions form one valid cuDF string
 * column.
 *
 * @tparam large_string_mode Whether the source string column uses 64-bit offsets.
 */
template <bool large_string_mode>
class string_compressed_sliced_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  static constexpr cudf::data_type offset_element_type = large_string_mode
                                                           ? cudf::data_type(cudf::type_id::INT64)
                                                           : cudf::data_type(cudf::type_id::INT32);

  using offsets_type = std::conditional_t<large_string_mode, int64_t, int32_t>;

  /**
   * @copydoc gqe::storage::output_column_builder::submit_materialization_requests()
   */
  void submit_materialization_requests(utility::copy_batch& c_batch,
                                       decompression_batch& d_batch,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) override;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream, rmm::device_async_resource_ref mr) override;

 private:
  /**
   * @brief Allocate character and offset output buffers.
   *
   * @param[in] char_buffer_size Number of bytes in the output character buffer.
   * @param[in] offset_buffer_size Number of bytes in the output offset buffer.
   * @param[in] stream CUDA stream used for allocation.
   * @return Character and offset output buffers.
   */
  [[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> allocate_output_buffers(
    size_t char_buffer_size,
    size_t offset_buffer_size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  rmm::device_buffer _char_buffer;
  rmm::device_buffer _offset_buffer;
  std::deque<rmm::device_buffer> _char_staging_buffers;
  std::deque<rmm::device_buffer> _offset_staging_buffers;
};

extern template class string_compressed_sliced_output_column_builder<false>;
extern template class string_compressed_sliced_output_column_builder<true>;

/**
 * @brief Builds a cuDF table from row groups selected by an in-memory read task.
 *
 * The builder chooses a concrete implementation of the abstract output_column_builder interface per
 * projected column, batches partition copies, runs required decompression, and constructs the final
 * output table.
 */
class output_table_builder {
 public:
  /**
   * @brief Construct an output table builder.
   *
   * @param[in] ctx_ref Execution context used for streams, memory resources, and semaphores.
   * @param[in] column_indexes Projected source column indices.
   * @param[in] data_types Expected cuDF data types for the projected columns.
   * @param[in] pruning_results Row groups and partitions selected by predicate pruning.
   */
  output_table_builder(context_reference ctx_ref,
                       std::vector<cudf::size_type> column_indexes,
                       std::vector<cudf::data_type> data_types,
                       std::unique_ptr<pruning_results_t> pruning_results);

  /**
   * @brief Materialize the selected row-group partitions as a cuDF table.
   *
   * @param[in] stream CUDA stream used for materialization.
   * @param[in] mr Memory resource used for device allocations.
   * @return The output table containing the projected columns and selected rows.
   */
  [[nodiscard]] std::unique_ptr<cudf::table> build(rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr);

 private:
  context_reference _ctx_ref;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;
  std::shared_ptr<pruning_results_t> _pruning_results;
  std::vector<std::unique_ptr<output_column_builder>> _output_columns;
};

/**
 * @brief A read task for loading data from memory.
 */
class in_memory_read_task : public read_task_base {
 public:
  /**
   * @brief Construct an in-memory read task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] row_groups The row groups assigned to this task.
   * @param[in] column_indexes Columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_indexes`.
   * @param[in] memory_kind The memory kind used by the input table.
   * @param[in] partial_filter Used to support predicate pushdown. Note that a row that satisfies
   * the predicate is guaranteed to be included in the loaded table, but a row that does not satisfy
   * the predicate may or may not be excluded. If such exclusion needs to be guaranteed, an extra
   * filter task is needed. If this argument is nullptr, no rows will be filtered out.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
   * @param[in] force_zero_copy_disable Override the zero-copy optimization parameter with a disable
   * command. Used, e.g., for unit testing or when zero-copy is unsafe due to the system
   * configuration.
   */
  in_memory_read_task(context_reference ctx_ref,
                      int32_t task_id,
                      int32_t stage_id,
                      std::vector<const row_group_interface*> row_groups,
                      std::vector<cudf::size_type> column_indexes,
                      std::vector<cudf::data_type> data_types,
                      memory_kind::type memory_kind,
                      std::optional<arrow::compute::Expression> partial_filter = std::nullopt,
                      std::vector<std::shared_ptr<task>> subquery_tasks        = {},
                      bool force_zero_copy_disable                             = false);

  in_memory_read_task(const in_memory_read_task&)            = delete;
  in_memory_read_task& operator=(const in_memory_read_task&) = delete;

  void execute() override;

 private:
  /// Determine the partitions of the table that should be emitted. If a partial filter exists,
  /// evaluate it on the zone map of on all row groups, and pair each row group with the evaluation
  /// result. If no partial filter exists, pair each row group with a partition which covers the
  /// entire row group.
  std::unique_ptr<pruning_results_t> evaluate_partial_filter();

  /// Determine if a zero-copy read is possible, based on the number of row group partitions and
  /// environment parameters.
  [[nodiscard]] bool is_zero_copy_possible(const pruning_results_t& pruning_results) const;

  /// Construct an empty table with the schema of the read task and emit it as an owned result.
  void emit_empty_table();

  /// Emit a single row group partition as a borrowed result, which is accessed via zero-copy.
  void emit_zero_copy_result(std::unique_ptr<pruning_results_t> pruning_results);

  /// Copy one or more row groups to GPU memory and emit an owned result.
  void emit_copied_result(std::unique_ptr<pruning_results_t> pruning_results);

  std::vector<const row_group_interface*>
    _row_groups; /**< Non-owning references to the row groups assigned to this task. */
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;
  memory_kind::type _memory_kind;
  std::optional<arrow::compute::Expression> _partial_filter;
  bool _force_zero_copy_disable;
};

/**
 * @brief A write task for storing data to an in-memory table.
 */
class in_memory_write_task : public write_task_base {
 public:
  using row_group_appender_t =
    std::variant<in_memory_table::row_group_appender, shared_in_memory_table::row_group_appender>;

  /**
   * @brief Construct an in-memory write task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input The input task emitting new data.
   * @param[in] non_owned_memory_resource Memory resource used to materialize columns.
   * @param[in] row_group_appender Destination table appender.
   * @param[in] memory_kind Memory kind of the destination table.
   * @param[in] column_indexes Destination index for each input column.
   * @param[in] data_types Expected input column types.
   * @param[in] statistics Statistics manager updated after a successful append.
   */
  in_memory_write_task(context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<task> input,
                       rmm::device_async_resource_ref non_owned_memory_resource,
                       row_group_appender_t row_group_appender,
                       memory_kind::type memory_kind,
                       std::vector<cudf::size_type> column_indexes,
                       std::vector<cudf::data_type> data_types,
                       table_statistics_manager* statistics);

  in_memory_write_task(const in_memory_write_task&)            = delete;
  in_memory_write_task& operator=(const in_memory_write_task&) = delete;

  void execute() override;

 private:
  rmm::device_async_resource_ref _non_owned_memory_resource;
  row_group_appender_t _row_group_appender;
  memory_kind::type _memory_kind;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;
  table_statistics_manager* _statistics;
};

/**
 * @brief Data access method to read an in-memory table.
 *
 * `get_read_tasks` acquires a shared read-only latch on the row groups
 * collection in the table for the duration of creating new read tasks.
 */
class in_memory_readable_view : public readable_view {
 public:
  friend class in_memory_table;
  friend class shared_in_memory_table;

  /**
   * @copydoc gqe::storage::readable_view::get_read_tasks()
   */
  std::vector<std::unique_ptr<read_task_base>> get_read_tasks(
    std::vector<readable_view::task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) override;

  /**
   * @copydoc gqe::storage::readable_view::get_iterate_task()
   */
  [[nodiscard]] std::unique_ptr<qep::iterate_task> get_iterate_task(
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types,
    std::optional<cudf::size_type> partition_column_idx = std::nullopt,
    std::unique_ptr<gqe::expression> partial_filter     = nullptr) override;

 private:
  in_memory_readable_view(in_memory_table_interface* non_owning_table);

  /// Transform a filter expression into an equivalent expression for the zone map.
  /// Returns the transformed expression; or std::nullopt, if:
  /// 1) the filter expression is nullptr
  /// 2) the filter expression cannot be transformed
  std::optional<arrow::compute::Expression> transform_partial_filter(
    gqe::expression* partial_filter);

  in_memory_table_interface* _non_owning_table;
};

/**
 * @brief Data access method to write an in-memory table.
 */
class in_memory_writeable_view : public writeable_view {
 public:
  friend in_memory_table;
  friend shared_in_memory_table;

  /**
   * @copydoc gqe::storage::writeable_view::get_write_tasks()
   */
  std::vector<std::unique_ptr<write_task_base>> get_write_tasks(
    std::vector<writeable_view::task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types,
    table_statistics_manager* statistics) override;

 private:
  in_memory_writeable_view(in_memory_table* non_owning_table);
  in_memory_writeable_view(shared_in_memory_table* non_owning_table);

  std::variant<in_memory_table*, shared_in_memory_table*> _non_owning_table;
};

}  // namespace storage

}  // namespace gqe
