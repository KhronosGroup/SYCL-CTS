/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "marray_common.h"

namespace marray_alignment {

using namespace sycl_cts;

// The marray type is passed using variable arguments due to it containing a
// comma.
#define ALIAS_TEST_IMPL(alias, ...) CHECK(std::is_same_v<__VA_ARGS__, alias>)

#define ALIAS_TEST_ELEMS(type, storage_type, elems) \
  ALIAS_TEST_IMPL(sycl::m##type##elems, sycl::marray<storage_type, elems>)

#define ALIAS_TEST(type, storage_type)     \
  ALIAS_TEST_ELEMS(type, storage_type, 2); \
  ALIAS_TEST_ELEMS(type, storage_type, 3); \
  ALIAS_TEST_ELEMS(type, storage_type, 4); \
  ALIAS_TEST_ELEMS(type, storage_type, 8); \
  ALIAS_TEST_ELEMS(type, storage_type, 16)

#define TEST_NAME marray_alignment

// have no type "mbool2", "mbool3", "mbool4", "mbool8", "mbool16"
TEST_CASE("marray_alignment", "[marray]") {
  ALIAS_TEST(char, int8_t);
  ALIAS_TEST(uchar, uint8_t);
  ALIAS_TEST(short, int16_t);
  ALIAS_TEST(ushort, uint16_t);
  ALIAS_TEST(int, int32_t);
  ALIAS_TEST(uint, uint32_t);
  ALIAS_TEST(long, int64_t);
  ALIAS_TEST(ulong, uint64_t);
  ALIAS_TEST(float, float);
  ALIAS_TEST(bool, bool);

  auto queue = util::get_cts_object::queue();

#if SYCL_CTS_ENABLE_HALF_TESTS
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations."
        "Skipping the test case.");
  } else {
    ALIAS_TEST(half, sycl::half);
  }
#endif  // SYCL_CTS_ENABLE_HALF_TESTS

#if SYCL_CTS_ENABLE_DOUBLE_TESTS
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations."
        "Skipping the test case.");
  } else {
    ALIAS_TEST(double, double);
  }
#endif  // SYCL_CTS_ENABLE_DOUBLE_TESTS
}

}  // namespace marray_alignment
