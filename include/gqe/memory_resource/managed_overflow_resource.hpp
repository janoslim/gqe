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

#pragma once

#include <gqe/utility/logger.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/managed_memory_resource.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief flowcsd: device query pool first, CUDA managed memory for what does not fit.
 *
 * Stock GQE fails a query with `rmm::out_of_memory` when its fixed device query pool is full.
 * This resource tries the pool first and, only when the pool throws `rmm::out_of_memory`, takes
 * the same request from CUDA managed memory, which the driver pages between GPU and host memory.
 * A query that fits in the pool therefore allocates exactly as in stock GQE.
 *
 * Every managed allocation is logged as one `flowcsd_managed_overflow` line (bytes of the request,
 * managed bytes live after it, and the peak so far), so a run's log says how much overflowed.
 */
class managed_overflow_resource {
 public:
  using device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

  explicit managed_overflow_resource(device_resource device_pool)
    : _state{std::make_shared<state>(std::move(device_pool))}
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    try {
      return _state->device_pool.allocate(stream, bytes, alignment);
    } catch (rmm::out_of_memory const&) {
      return _state->allocate_managed(bytes);
    }
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (_state->release_managed(ptr, bytes)) { return; }
    _state->device_pool.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    try {
      return _state->device_pool.allocate_sync(bytes, alignment);
    } catch (rmm::out_of_memory const&) {
      return _state->allocate_managed(bytes);
    }
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (_state->release_managed(ptr, bytes)) { return; }
    _state->device_pool.deallocate_sync(ptr, bytes, alignment);
  }

  friend void get_property(managed_overflow_resource const&, cuda::mr::device_accessible) noexcept
  {
  }

  [[nodiscard]] bool operator==(managed_overflow_resource const& other) const noexcept
  {
    return _state == other._state;
  }
  [[nodiscard]] bool operator!=(managed_overflow_resource const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  // Shared so that copies made by cuda::mr::any_resource see one pool and one set of managed
  // pointers.
  struct state {
    explicit state(device_resource pool) : device_pool{std::move(pool)} {}

    void* allocate_managed(std::size_t bytes)
    {
      // cudaMallocManaged returns 256-byte aligned memory, which covers every alignment RMM asks
      // for (rmm::CUDA_ALLOCATION_ALIGNMENT).
      void* ptr = managed.allocate_sync(bytes);
      std::lock_guard lock{mutex};
      managed_pointers.insert(ptr);
      managed_bytes += bytes;
      peak_managed_bytes = std::max(peak_managed_bytes, managed_bytes);
      GQE_LOG_INFO("flowcsd_managed_overflow bytes={} managed_bytes={} peak_managed_bytes={}",
                   bytes,
                   managed_bytes,
                   peak_managed_bytes);
      return ptr;
    }

    // Frees `ptr` if it is a managed allocation; false when it came from the device pool.
    bool release_managed(void* ptr, std::size_t bytes) noexcept
    {
      {
        std::lock_guard lock{mutex};
        if (managed_pointers.erase(ptr) == 0) { return false; }
        managed_bytes -= bytes;
      }
      managed.deallocate_sync(ptr, bytes);
      return true;
    }

    device_resource device_pool;
    rmm::mr::managed_memory_resource managed;
    std::mutex mutex;
    std::unordered_set<void*> managed_pointers;
    std::size_t managed_bytes      = 0;
    std::size_t peak_managed_bytes = 0;
  };

  std::shared_ptr<state> _state;
};

}  // namespace memory_resource

}  // namespace gqe
