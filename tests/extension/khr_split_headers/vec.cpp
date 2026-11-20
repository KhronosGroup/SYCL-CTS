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
#include <sycl/khr/split_headers/vec.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the vec header defines the vec class template",
          "[khr_split_headers][vec]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::vec<int, 1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::vec<int, 2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::vec<int, 3>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::vec<int, 4>>);
}

}  // namespace khr_split_headers::tests

#endif  // SYCL_KHR_SPLIT_HEADERS
