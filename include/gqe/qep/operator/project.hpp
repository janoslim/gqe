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
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Evaluate projection expressions on a regular or masked table.
 *
 * Column-reference expressions pass the input column state through. Computed expressions produce
 * owned columns.
 *
 * Before evaluating projection expressions that are not just column references, project creates
 * temporary views of any input columns they reference. Rows rejected by the valid mask are null in
 * those views. The original input columns are unchanged, and the valid-mask state is passed through
 * to the output.
 */
class project_task : public optional_transform_task {
 public:
  /**
   * @brief Construct a projection task.
   *
   * @param[in] output_expressions Projection expressions in output-column order.
   */
  explicit project_task(std::vector<std::unique_ptr<gqe::expression>> output_expressions);

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
  std::vector<std::unique_ptr<gqe::expression>> _output_expressions;
};

}  // namespace qep
}  // namespace gqe
