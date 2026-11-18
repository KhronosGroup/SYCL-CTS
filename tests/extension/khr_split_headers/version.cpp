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
#include <sycl/khr/split_headers/version.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the version header defines the SYCL_LANGUAGE_VERSION macro",
          "[khr_split_headers][version]") {
#ifdef SYCL_LANGUAGE_VERSION
  constexpr unsigned long long version = SYCL_LANGUAGE_VERSION;
#else
  constexpr unsigned long long version = 0;
#endif
  STATIC_REQUIRE(version == 202012L);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the version tests require sycl_khr_split_headers",
          "[khr_split_headers][version]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
