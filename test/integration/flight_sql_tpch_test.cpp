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

#include "flight_sql_test_harness.hpp"
#include "flight_sql_test_utils.hpp"
#include "tpch_query_sets.hpp"

#include <gtest/gtest.h>

#include <format>
#include <tuple>

namespace gqe_test {

// ---------------------------------------------------------------------------
// TPC-H query test — parameterized by query number.
// ---------------------------------------------------------------------------

class TpchQueryTest : public ::testing::TestWithParam<std::tuple<ServerConfiguration, int>> {
 protected:
  void SetUp() override
  {
    if (!g_config.data_path || !g_config.queries_dir || !g_config.ref_results) {
      GTEST_SKIP() << "TPCH_DATA_PATH, TPCH_QUERIES, and TPCH_REF_RESULTS must be set";
    }
    auto const& server_config = std::get<0>(GetParam());
    if (server_config.num_gpus > g_config.host_gpu_count) {
      GTEST_SKIP() << "Test requires " << server_config.num_gpus << " GPUs but host has "
                   << g_config.host_gpu_count;
    }
    ASSERT_TRUE(g_server.start(server_config,
                               g_config.node_manager_bin,
                               g_config.task_manager_bin,
                               g_config.load_script,
                               g_config.data_path))
      << "Failed to start server config " << server_config.name << ": " << g_server.load_error();
    ASSERT_TRUE(g_server.ready()) << "Server is not ready";
  }
};

TEST_P(TpchQueryTest, ExecuteQuery)
{
  auto const query_num     = std::get<1>(GetParam());
  auto [exit_code, output] = run_tpch_query(g_config.run_script,
                                            InputFormat::Sql,
                                            g_server.port(),
                                            g_config.queries_dir,
                                            query_num,
                                            g_config.ref_results);
  ASSERT_EQ(exit_code, 0) << "TPC-H Q" << query_num << " failed. Output:\n" << output;
}

INSTANTIATE_TEST_SUITE_P(
  LocalContiguousTableSingleGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kLocalContiguousTableSingleGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

INSTANTIATE_TEST_SUITE_P(
  LocalContiguousTableMultiGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kLocalContiguousTableMultiGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

INSTANTIATE_TEST_SUITE_P(
  SharedContiguousTableSingleGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kSharedContiguousTableSingleGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

INSTANTIATE_TEST_SUITE_P(
  SharedCompressedTableSingleGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kSharedCompressedTableSingleGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

}  // namespace gqe_test
