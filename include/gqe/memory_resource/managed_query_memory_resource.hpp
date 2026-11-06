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

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <rmm/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief flowcsd: marks the calling thread as running a Parquet scan's reader.
 *
 * While a scope is alive, the thread's allocations through `managed_query_memory_resource` come
 * from the device scan-buffer pool. cuDF's Parquet reader allocates the buffers cuFile reads into
 * (and its other temporaries) from the current device resource, and cuFile cannot DMA into CUDA
 * managed memory, so those buffers must stay in device memory.
 */
class scan_buffer_scope {
 public:
  scan_buffer_scope() noexcept : _previous{active} { active = true; }
  ~scan_buffer_scope() { active = _previous; }
  scan_buffer_scope(scan_buffer_scope const&)            = delete;
  scan_buffer_scope& operator=(scan_buffer_scope const&) = delete;

  [[nodiscard]] static bool is_active() noexcept { return active; }

 private:
  static inline thread_local bool active = false;
  bool _previous;
};

/**
 * @brief flowcsd: the query memory of GQE on CUDA managed memory, as Sirius's managed_memory setting.
 *
 * Stock GQE serves every query allocation from a fixed device pool and fails the query when it is
 * full. This resource serves them from a pool on CUDA managed memory instead, which the driver
 * pages between GPU and host memory, except allocations made inside a `scan_buffer_scope`, which
 * come from the device scan-buffer pool. Frees go back to the pool the pointer came from.
 */
class managed_query_memory_resource {
 public:
  using device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

  managed_query_memory_resource(device_resource scan_buffer_pool, device_resource managed_pool)
    : _state{std::make_shared<state>(std::move(scan_buffer_pool), std::move(managed_pool))}
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (!scan_buffer_scope::is_active()) {
      return _state->managed_pool.allocate(stream, bytes, alignment);
    }
    void* ptr = _state->scan_buffer_pool.allocate(stream, bytes, alignment);
    _state->remember_scan_buffer(ptr);
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (_state->forget_scan_buffer(ptr)) {
      _state->scan_buffer_pool.deallocate(stream, ptr, bytes, alignment);
      return;
    }
    _state->managed_pool.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (!scan_buffer_scope::is_active()) {
      return _state->managed_pool.allocate_sync(bytes, alignment);
    }
    void* ptr = _state->scan_buffer_pool.allocate_sync(bytes, alignment);
    _state->remember_scan_buffer(ptr);
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (_state->forget_scan_buffer(ptr)) {
      _state->scan_buffer_pool.deallocate_sync(ptr, bytes, alignment);
      return;
    }
    _state->managed_pool.deallocate_sync(ptr, bytes, alignment);
  }

  /**
   * @brief The managed pool itself, for the tables a scan returns (they outlive the scan and must
   * not occupy the scan-buffer pool).
   */
  [[nodiscard]] rmm::device_async_resource_ref managed_pool() const noexcept
  {
    return rmm::device_async_resource_ref{_state->managed_pool};
  }

  friend void get_property(managed_query_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  [[nodiscard]] bool operator==(managed_query_memory_resource const& other) const noexcept
  {
    return _state == other._state;
  }
  [[nodiscard]] bool operator!=(managed_query_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  // Shared so that copies made by cuda::mr::any_resource see the same pools and pointer set.
  struct state {
    state(device_resource scan_pool, device_resource managed)
      : scan_buffer_pool{std::move(scan_pool)}, managed_pool{std::move(managed)}
    {
    }

    void remember_scan_buffer(void* ptr)
    {
      std::lock_guard lock{mutex};
      scan_buffers.insert(ptr);
    }

    // True when `ptr` came from the scan-buffer pool (and forgets it).
    bool forget_scan_buffer(void* ptr) noexcept
    {
      std::lock_guard lock{mutex};
      return scan_buffers.erase(ptr) == 1;
    }

    device_resource scan_buffer_pool;
    device_resource managed_pool;
    std::mutex mutex;
    std::unordered_set<void*> scan_buffers;
  };

  std::shared_ptr<state> _state;
};

/**
 * @brief The resource for the tables a Parquet scan returns: the managed pool of the task
 * manager's `managed_query_memory_resource`, set once when the task manager context is built.
 */
inline std::optional<rmm::device_async_resource_ref>& scan_output_resource()
{
  static std::optional<rmm::device_async_resource_ref> resource;
  return resource;
}

}  // namespace memory_resource

}  // namespace gqe
