/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2026 The Khronos Group Inc.
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

// The sycl_khr_max_work_group_queries extension states that when an
// implementation also implements sycl_khr_split_headers, the APIs defined in
// the sycl::khr namespace are additionally provided by the header
// <sycl/khr/max_work_group_queries.hpp>, and the feature test macro is
// additionally provided by <sycl/khr/split_headers/version.hpp>. These tests
// verify that coexistence and are guarded on both SYCL_KHR_SPLIT_HEADERS and
// SYCL_KHR_MAX_WORK_GROUP_QUERIES.

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/version.hpp>

#if defined(SYCL_KHR_SPLIT_HEADERS) && defined(SYCL_KHR_MAX_WORK_GROUP_QUERIES)

#include <sycl/khr/max_work_group_queries.hpp>
#include <sycl/khr/split_headers/device.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE(
    "the max_work_group_queries descriptors are provided through "
    "<sycl/khr/max_work_group_queries.hpp> when split_headers is implemented",
    "[khr_split_headers][max_work_group_queries]") {
  // The khr::info::device descriptors introduced by the extension are complete.
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::khr::info::device::max_work_group_range>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::khr::info::device::max_work_group_range_size>);
}

TEST_CASE(
    "the max_work_group_queries feature test macro is provided through "
    "<sycl/khr/split_headers/version.hpp>",
    "[khr_split_headers][max_work_group_queries]") {
  STATIC_REQUIRE(SYCL_KHR_MAX_WORK_GROUP_QUERIES >= 1);
}

}  // namespace khr_split_headers::tests

#else

TEST_CASE(
    "max_work_group_queries split_headers coexistence is not applicable",
    "[khr_split_headers][max_work_group_queries]") {
  SUCCEED(
      "SYCL_KHR_SPLIT_HEADERS and/or SYCL_KHR_MAX_WORK_GROUP_QUERIES not "
      "implemented");
}

#endif
