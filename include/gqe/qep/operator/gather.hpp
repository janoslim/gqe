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

#include <gqe/qep/task.hpp>

#include <memory>
#include <optional>

namespace gqe {
namespace qep {

/**
 * @brief Apply an optional valid mask and materialize indirect columns.
 *
 * Accepts regular tables, count-only state, masked tables, and indirect tables with an optional
 * valid mask. Indirect columns that share a gather source are materialized by one gather operation.
 * Regular-table and count-only inputs pass through without materialization. For an unmasked
 * indirect table, existing `cudf_column` and `cudf_column_view` states also pass through, while
 * each `indirect_column` state is materialized as a `cudf_column` state.
 */
class gather_task : public optional_transform_task {
 public:
  /** @copydoc optional_transform_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;
};

}  // namespace qep
}  // namespace gqe
