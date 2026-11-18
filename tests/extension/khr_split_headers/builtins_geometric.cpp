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
#include <sycl/khr/split_headers/builtins_geometric.hpp>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/builtins_geometric.hpp>
// provides the geometric functions from the SYCL specification. We use scalar
// float inputs (a scalar float is a "generic geometric type" and a "float
// geometric type") so the header can be validated standalone, without pulling
// in vec/marray headers. For scalar inputs each function returns the floating
// point argument type. The cross function is not tested here because it is only
// defined for 3- or 4-element vec/marray inputs.

namespace khr_split_headers::tests {

TEST_CASE("the builtins_geometric header defines the dot function",
          "[khr_split_headers][builtins_geometric]") {
  using return_t =
      decltype(sycl::dot(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_geometric header defines the distance function",
          "[khr_split_headers][builtins_geometric]") {
  using return_t =
      decltype(sycl::distance(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_geometric header defines the length function",
          "[khr_split_headers][builtins_geometric]") {
  using return_t = decltype(sycl::length(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_geometric header defines the normalize function",
          "[khr_split_headers][builtins_geometric]") {
  using return_t = decltype(sycl::normalize(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_geometric header defines the fast_distance function",
          "[khr_split_headers][builtins_geometric]") {
  using return_t = decltype(sycl::fast_distance(std::declval<float>(),
                                                std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_geometric header defines the fast_length function",
          "[khr_split_headers][builtins_geometric]") {
  using return_t = decltype(sycl::fast_length(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_geometric header defines the fast_normalize function",
          "[khr_split_headers][builtins_geometric]") {
  using return_t = decltype(sycl::fast_normalize(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_geometric header provides functions for double scalars",
          "[khr_split_headers][builtins_geometric]") {
  using length_t = decltype(sycl::length(std::declval<double>()));
  using dot_t =
      decltype(sycl::dot(std::declval<double>(), std::declval<double>()));
  STATIC_REQUIRE(std::is_same_v<length_t, double>);
  STATIC_REQUIRE(std::is_same_v<dot_t, double>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the builtins_geometric tests require sycl_khr_split_headers",
          "[khr_split_headers][builtins_geometric]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
