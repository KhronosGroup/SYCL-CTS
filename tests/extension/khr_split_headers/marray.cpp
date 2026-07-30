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
#include <sycl/khr/split_headers/marray.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the marray header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][marray]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the marray header defines the marray class template",
          "[khr_split_headers][marray]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 3>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::marray<int, 4>>);
}

}  // namespace khr_split_headers::tests
