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

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/groups.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the groups header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][groups]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the groups header defines the group class template",
          "[khr_split_headers][groups]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::group<1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::group<2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::group<3>>);
}

TEST_CASE("the groups header defines the sub_group class",
          "[khr_split_headers][groups]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::sub_group>);
}

TEST_CASE("the groups header defines the device_event class",
          "[khr_split_headers][groups]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::device_event>);
}

}  // namespace khr_split_headers::tests
