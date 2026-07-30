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
#include <sycl/khr/split_headers/builtins_relational.hpp>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/builtins_relational.hpp>
// provides the relational functions from the SYCL specification. We use scalar
// inputs (plain float/double/int) so the header can be validated standalone,
// without pulling in vec/marray headers. For scalar floating point inputs the
// comparison functions return bool; bitselect and select return their operand
// type.

namespace khr_split_headers::tests {

TEST_CASE(
    "the builtins_relational header defines the SYCL_KHR_SPLIT_HEADERS macro",
    "[khr_split_headers][builtins_relational]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the builtins_relational header defines the isequal function",
          "[khr_split_headers][builtins_relational]") {
  using return_t =
      decltype(sycl::isequal(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isnotequal function",
          "[khr_split_headers][builtins_relational]") {
  using return_t =
      decltype(sycl::isnotequal(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isgreater function",
          "[khr_split_headers][builtins_relational]") {
  using return_t =
      decltype(sycl::isgreater(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isgreaterequal function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::isgreaterequal(std::declval<float>(),
                                                 std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isless function",
          "[khr_split_headers][builtins_relational]") {
  using return_t =
      decltype(sycl::isless(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the islessequal function",
          "[khr_split_headers][builtins_relational]") {
  using return_t =
      decltype(sycl::islessequal(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the islessgreater function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::islessgreater(std::declval<float>(),
                                                std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isfinite function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::isfinite(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isinf function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::isinf(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isnan function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::isnan(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isnormal function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::isnormal(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isordered function",
          "[khr_split_headers][builtins_relational]") {
  using return_t =
      decltype(sycl::isordered(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the isunordered function",
          "[khr_split_headers][builtins_relational]") {
  using return_t =
      decltype(sycl::isunordered(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the signbit function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::signbit(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the bitselect function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::bitselect(
      std::declval<int>(), std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_relational header defines the select function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::select(
      std::declval<int>(), std::declval<int>(), std::declval<bool>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

// any and all only provide scalar overloads through the deprecated SYCL 2020
// interface; the non-deprecated overloads require marray/vec inputs.
TEST_CASE("the builtins_relational header defines the any function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::any(std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the builtins_relational header defines the all function",
          "[khr_split_headers][builtins_relational]") {
  using return_t = decltype(sycl::all(std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE(
    "the builtins_relational header provides functions for double scalars",
    "[khr_split_headers][builtins_relational]") {
  using isnan_t = decltype(sycl::isnan(std::declval<double>()));
  using isequal_t =
      decltype(sycl::isequal(std::declval<double>(), std::declval<double>()));
  STATIC_REQUIRE(std::is_same_v<isnan_t, bool>);
  STATIC_REQUIRE(std::is_same_v<isequal_t, bool>);
}

}  // namespace khr_split_headers::tests
