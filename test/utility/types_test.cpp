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

#include <gqe/types.hpp>

#include <gtest/gtest.h>

// Test code must live in the global namespace (repo guideline); bring gqe names into scope.
using namespace gqe;

TEST(TypesTest, IoEngineTypeRoundTrip)
{
  for (auto engine : {io_engine_type::io_uring, io_engine_type::psync, io_engine_type::automatic}) {
    EXPECT_EQ(from_string<io_engine_type>(to_string(engine)), engine);
  }
}

TEST(TypesTest, CompressionFormatRoundTrip)
{
  for (auto fmt : {compression_format::none,
                   compression_format::ans,
                   compression_format::lz4,
                   compression_format::snappy,
                   compression_format::gdeflate,
                   compression_format::deflate,
                   compression_format::cascaded,
                   compression_format::bitpacking,
                   compression_format::zstd,
                   compression_format::gzip,
                   compression_format::bitcomp}) {
    EXPECT_EQ(from_string<compression_format>(to_string(fmt)), fmt);
  }
}

TEST(TypesTest, DecompressionBackendRoundTrip)
{
  for (auto backend :
       {decompression_backend::auto_, decompression_backend::sm, decompression_backend::de}) {
    EXPECT_EQ(from_string<decompression_backend>(to_string(backend)), backend);
  }
}

TEST(TypesTest, DecimalTypeRoundTrip)
{
  for (auto dt : {decimal_type::double_, decimal_type::decimal}) {
    EXPECT_EQ(from_string<decimal_type>(to_string(dt)), dt);
  }
}

TEST(TypesTest, StorageKindIsFileBacked)
{
  // parquet_file is the only file-backed kind.
  EXPECT_TRUE(storage_kind::is_file_backed(storage_kind::parquet_file{{"f.parquet"}}));

  // Every memory-backed kind is not file-backed.
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::system_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::numa_memory{cpu_set{}}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::pinned_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::device_memory{rmm::cuda_device_id{0}}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::managed_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::numa_pinned_memory{cpu_set{}}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::boost_shared_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::numa_pool_memory{0}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::shared_numa_pool_memory{0}));
}
