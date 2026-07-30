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
#include <sycl/khr/split_headers/span.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the span header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][span]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the span header defines the span class template",
          "[khr_split_headers][span]") {
  using span_t = sycl::span<int, 1>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<span_t>);
}

TEST_CASE("the span header defines the dynamic_extent constant",
          "[khr_split_headers][span]") {
  STATIC_REQUIRE(
      std::is_same_v<decltype(sycl::dynamic_extent), const std::size_t>);
}

}  // namespace khr_split_headers::tests
