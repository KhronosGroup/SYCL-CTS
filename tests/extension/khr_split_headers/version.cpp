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
#include <sycl/khr/split_headers/version.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the version header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][version]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

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
