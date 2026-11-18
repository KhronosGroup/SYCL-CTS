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
#include <sycl/khr/split_headers/marray.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the marray header defines the marray class template",
          "[khr_split_headers][marray]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 3>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 4>>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the marray tests require sycl_khr_split_headers",
          "[khr_split_headers][marray]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
