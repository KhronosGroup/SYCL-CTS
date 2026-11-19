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

namespace khr_split_headers::tests {

// This test case is the only one of the category that does not depend on any
// header that the extension introduces, so it is the one that reports the
// category as skipped when the extension is not implemented.
TEST_CASE(
    "the split headers extension defines the SYCL_KHR_SPLIT_HEADERS macro",
    "[khr_split_headers][version]") {
#if SYCL_KHR_SPLIT_HEADERS
  STATIC_REQUIRE(SYCL_KHR_SPLIT_HEADERS == 1);
#else
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
#endif
}

#if SYCL_KHR_SPLIT_HEADERS

TEST_CASE("the version header defines the SYCL_LANGUAGE_VERSION macro",
          "[khr_split_headers][version]") {
#ifdef SYCL_LANGUAGE_VERSION
  constexpr unsigned long long version = SYCL_LANGUAGE_VERSION;
#else
  constexpr unsigned long long version = 0;
#endif
  STATIC_REQUIRE(version == 202012L);
}

#endif  // SYCL_KHR_SPLIT_HEADERS

}  // namespace khr_split_headers::tests
