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
#include <sycl/khr/split_headers/index_space.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the index_space header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][index_space]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the index_space header defines the id class template",
          "[khr_split_headers][index_space]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::id<1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::id<2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::id<3>>);
}

TEST_CASE("the index_space header defines the item class template",
          "[khr_split_headers][index_space]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::item<1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::item<2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::item<3>>);
}

TEST_CASE("the index_space header defines the nd_item class template",
          "[khr_split_headers][index_space]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::nd_item<1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::nd_item<2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::nd_item<3>>);
}

TEST_CASE("the index_space header defines the range class template",
          "[khr_split_headers][index_space]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::range<1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::range<2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::range<3>>);
}

TEST_CASE("the index_space header defines the nd_range class template",
          "[khr_split_headers][index_space]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::nd_range<1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::nd_range<2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::nd_range<3>>);
}

}  // namespace khr_split_headers::tests
