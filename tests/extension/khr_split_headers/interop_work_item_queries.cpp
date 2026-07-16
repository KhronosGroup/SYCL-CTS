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

// The sycl_khr_work_item_queries extension states that when an implementation
// also implements sycl_khr_split_headers, the APIs defined in the sycl::khr
// namespace are additionally provided by the header
// <sycl/khr/work_item_queries.hpp>, and the feature test macro is additionally
// provided by <sycl/khr/split_headers/version.hpp>. These tests verify that
// coexistence and are guarded on both SYCL_KHR_SPLIT_HEADERS and
// SYCL_KHR_WORK_ITEM_QUERIES.

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/version.hpp>

#if defined(SYCL_KHR_SPLIT_HEADERS) && defined(SYCL_KHR_WORK_ITEM_QUERIES)

#include <sycl/khr/split_headers/index_space.hpp>
#include <sycl/khr/work_item_queries.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE(
    "the work_item_queries APIs are provided through "
    "<sycl/khr/work_item_queries.hpp> when split_headers is implemented",
    "[khr_split_headers][work_item_queries]") {
  // khr::this_nd_item / this_group return the corresponding index-space types.
  using nd_item_t = decltype(sycl::khr::this_nd_item<1>());
  using group_t = decltype(sycl::khr::this_group<1>());
  using sub_group_t = decltype(sycl::khr::this_sub_group());
  STATIC_REQUIRE(std::is_same_v<nd_item_t, sycl::nd_item<1>>);
  STATIC_REQUIRE(std::is_same_v<group_t, sycl::group<1>>);
  STATIC_REQUIRE(std::is_same_v<sub_group_t, sycl::sub_group>);
}

TEST_CASE(
    "the work_item_queries feature test macro is provided through "
    "<sycl/khr/split_headers/version.hpp>",
    "[khr_split_headers][work_item_queries]") {
  STATIC_REQUIRE(SYCL_KHR_WORK_ITEM_QUERIES >= 1);
}

}  // namespace khr_split_headers::tests

#else

TEST_CASE("work_item_queries split_headers coexistence is not applicable",
          "[khr_split_headers][work_item_queries]") {
  SUCCEED(
      "SYCL_KHR_SPLIT_HEADERS and/or SYCL_KHR_WORK_ITEM_QUERIES not "
      "implemented");
}

#endif
