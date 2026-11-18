/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "util.h"
#include <catch2/catch_test_macros.hpp>

#if SYCL_KHR_SPLIT_HEADERS
#include <sycl/khr/split_headers/index_space.hpp>

namespace khr_split_headers::tests {

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

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the index_space tests require sycl_khr_split_headers",
          "[khr_split_headers][index_space]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
