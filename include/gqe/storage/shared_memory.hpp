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

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include <cstddef>

namespace gqe {

/// @brief Allocator for metadata stored in Boost managed shared memory.
using shared_memory_allocator =
  boost::interprocess::allocator<std::byte,
                                 boost::interprocess::managed_shared_memory::segment_manager>;

/// @brief Numeric handle for objects stored in Boost managed shared memory.
using shared_memory_handle_t = boost::interprocess::managed_shared_memory::handle_t;

}  // namespace gqe
