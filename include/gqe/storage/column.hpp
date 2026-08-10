/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/storage/buffer.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace gqe {

/**
 * @brief Allocator-aware recursive column storage.
 *
 * @tparam Allocator Allocator used for persistent host-side child metadata.
 */
template <class Allocator = std::allocator<std::byte>>
class column {
 public:
  using allocator_type = gqe::storage_allocator_t<Allocator>;
  template <class T>
  using vector_type = gqe::storage_vector_t<Allocator, T>;

  column() : _children(gqe::storage_rebind_allocator_t<Allocator, column>(allocator_type{})) {}

  column(cudf::column_view source,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr,
         allocator_type const& alloc = allocator_type{})
    : _type(source.type()),
      _size(source.size()),
      _null_count(source.null_count()),
      _children(gqe::storage_rebind_allocator_t<Allocator, column>(alloc))
  {
    std::size_t data_buffer_size = 0;
    if (source.data<std::byte>() != nullptr && source.size() != 0) {
      if (source.type().id() == cudf::type_id::STRING) {
        data_buffer_size =
          static_cast<std::size_t>(cudf::strings_column_view(source).chars_size(stream));
      } else if (cudf::is_fixed_width(source.type())) {
        data_buffer_size = static_cast<std::size_t>(source.size()) *
                           static_cast<std::size_t>(cudf::size_of(source.type()));
      }
    }
    if (data_buffer_size > 0) {
      _data.allocate(data_buffer_size, stream, mr);
      GQE_CUDA_TRY(cudaMemcpyAsync(_data.data(),
                                   source.data<std::byte>(),
                                   data_buffer_size,
                                   cudaMemcpyDefault,
                                   stream.value()));
    }

    auto const null_mask_buffer_size =
      source.null_count() == 0 || source.null_mask() == nullptr || source.size() == 0
        ? 0
        : static_cast<std::size_t>(cudf::num_bitmask_words(source.size())) *
            sizeof(cudf::bitmask_type);
    if (null_mask_buffer_size > 0) {
      _null_mask.allocate(null_mask_buffer_size, stream, mr);
      GQE_CUDA_TRY(cudaMemcpyAsync(_null_mask.data(),
                                   source.null_mask(),
                                   null_mask_buffer_size,
                                   cudaMemcpyDefault,
                                   stream.value()));
    }

    _children.reserve(source.num_children());
    for (auto idx = 0; idx < source.num_children(); ++idx) {
      _children.emplace_back(source.child(idx), stream, mr, alloc);
    }
  }

  ~column() = default;

  column(column const&)                = delete;
  column& operator=(column const&)     = delete;
  column(column&&) noexcept            = default;
  column& operator=(column&&) noexcept = default;

  [[nodiscard]] cudf::column_view view() const
  {
    std::vector<cudf::column_view> child_views(_children.size());
    for (std::size_t idx = 0; idx < _children.size(); ++idx) {
      child_views[idx] = _children[idx].view();
    }

    auto const* null_mask =
      _null_count > 0 ? static_cast<cudf::bitmask_type const*>(_null_mask.data()) : nullptr;
    return cudf::column_view(
      _type, _size, static_cast<void const*>(_data.data()), null_mask, _null_count, 0, child_views);
  }

  [[nodiscard]] cudf::mutable_column_view mutable_view()
  {
    std::vector<cudf::mutable_column_view> child_views(_children.size());
    for (std::size_t idx = 0; idx < _children.size(); ++idx) {
      child_views[idx] = _children[idx].mutable_view();
    }

    auto* null_mask =
      _null_count > 0 ? static_cast<cudf::bitmask_type*>(_null_mask.data()) : nullptr;
    return cudf::mutable_column_view(
      _type, _size, static_cast<void*>(_data.data()), null_mask, _null_count, 0, child_views);
  }

  [[nodiscard]] cudf::data_type type() const noexcept { return _type; }
  [[nodiscard]] cudf::size_type size() const noexcept { return _size; }
  [[nodiscard]] cudf::size_type null_count() const noexcept { return _null_count; }
  [[nodiscard]] std::size_t data_size() const noexcept { return _data.size(); }
  [[nodiscard]] std::size_t null_mask_size() const noexcept { return _null_mask.size(); }

  [[nodiscard]] std::size_t total_data_size() const noexcept
  {
    std::size_t total = data_size() + null_mask_size();
    for (auto const& child : _children) {
      total += child.total_data_size();
    }
    return total;
  }

  [[nodiscard]] column const& child(cudf::size_type index) const
  {
    assert(index >= 0);
    return _children.at(static_cast<std::size_t>(index));
  }

  [[nodiscard]] column& child(cudf::size_type index)
  {
    assert(index >= 0);
    return _children.at(static_cast<std::size_t>(index));
  }

  [[nodiscard]] cudf::size_type num_children() const noexcept
  {
    assert(_children.size() <=
           static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));
    return static_cast<cudf::size_type>(_children.size());
  }

  void deallocate(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    for (auto& child : _children) {
      child.deallocate(stream, mr);
    }
    _children.clear();
    _data.deallocate(stream, mr);
    _null_mask.deallocate(stream, mr);
    _type       = cudf::data_type(cudf::type_id::EMPTY);
    _size       = 0;
    _null_count = 0;
  }

 private:
  cudf::data_type _type{cudf::type_id::EMPTY};
  cudf::size_type _size{};
  cudf::size_type _null_count{};
  gqe::storage::buffer _data;
  gqe::storage::buffer _null_mask;
  vector_type<column> _children;
};

}  // namespace gqe
