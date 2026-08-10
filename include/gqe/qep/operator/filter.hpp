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

#include <gqe/expression/expression.hpp>
#include <gqe/qep/task.hpp>

#include <memory>
#include <optional>

namespace gqe {
namespace qep {

/**
 * @brief Evaluate a Boolean predicate and attach it to the input as a valid-row mask.
 *
 * Accepts either `[columns...]` or `[valid_mask, columns...]`. The output is always a masked
 * table and passes the original data-column states through without copying.
 */
class filter_task : public optional_transform_task {
 public:
  /**
   * @brief Construct a filter task.
   *
   * @param[in] condition Boolean predicate used to produce the valid-row mask.
   */
  explicit filter_task(std::unique_ptr<gqe::expression> condition);

  /**
   * @copydoc optional_transform_task::next
   */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::unique_ptr<gqe::expression> _condition;
};

}  // namespace qep
}  // namespace gqe
