/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2026 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sycl/khr/split_headers/builtins_integer.hpp>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/builtins_integer.hpp>
// provides the integer functions from the SYCL specification. We use scalar
// "generic integer types" (plain C++ integer types) so that the header can be
// validated standalone, without pulling in vec/marray headers. Each function's
// return type equals its (first) argument type for scalar inputs.

namespace khr_split_headers::tests {

TEST_CASE(
    "the builtins_integer header defines the SYCL_KHR_SPLIT_HEADERS macro",
    "[khr_split_headers][builtins_integer]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the builtins_integer header defines the abs function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::abs(std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the abs_diff function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::abs_diff(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the add_sat function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::add_sat(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the hadd function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::hadd(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the rhadd function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::rhadd(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the clamp function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::clamp(
      std::declval<int>(), std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the clz function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::clz(std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the ctz function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::ctz(std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the mad_hi function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::mad_hi(
      std::declval<int>(), std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the mad_sat function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::mad_sat(
      std::declval<int>(), std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the max function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::max(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the min function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::min(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the mul_hi function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::mul_hi(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the rotate function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::rotate(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the sub_sat function",
          "[khr_split_headers][builtins_integer]") {
  using return_t =
      decltype(sycl::sub_sat(std::declval<int>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the popcount function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::popcount(std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the builtins_integer header defines the mad24 function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::mad24(std::declval<std::int32_t>(),
                                        std::declval<std::int32_t>(),
                                        std::declval<std::int32_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, std::int32_t>);
}

TEST_CASE("the builtins_integer header defines the mul24 function",
          "[khr_split_headers][builtins_integer]") {
  using return_t = decltype(sycl::mul24(std::declval<std::int32_t>(),
                                        std::declval<std::int32_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, std::int32_t>);
}

TEST_CASE("the builtins_integer header defines the unsigned upsample overloads",
          "[khr_split_headers][builtins_integer]") {
  using ret16 = decltype(sycl::upsample(std::declval<std::uint8_t>(),
                                        std::declval<std::uint8_t>()));
  using ret32 = decltype(sycl::upsample(std::declval<std::uint16_t>(),
                                        std::declval<std::uint16_t>()));
  using ret64 = decltype(sycl::upsample(std::declval<std::uint32_t>(),
                                        std::declval<std::uint32_t>()));
  STATIC_REQUIRE(std::is_same_v<ret16, std::uint16_t>);
  STATIC_REQUIRE(std::is_same_v<ret32, std::uint32_t>);
  STATIC_REQUIRE(std::is_same_v<ret64, std::uint64_t>);
}

TEST_CASE("the builtins_integer header defines the signed upsample overloads",
          "[khr_split_headers][builtins_integer]") {
  using ret16 = decltype(sycl::upsample(std::declval<std::int8_t>(),
                                        std::declval<std::uint8_t>()));
  using ret32 = decltype(sycl::upsample(std::declval<std::int16_t>(),
                                        std::declval<std::uint16_t>()));
  using ret64 = decltype(sycl::upsample(std::declval<std::int32_t>(),
                                        std::declval<std::uint32_t>()));
  STATIC_REQUIRE(std::is_same_v<ret16, std::int16_t>);
  STATIC_REQUIRE(std::is_same_v<ret32, std::int32_t>);
  STATIC_REQUIRE(std::is_same_v<ret64, std::int64_t>);
}

TEST_CASE(
    "the builtins_integer header provides functions for unsigned scalar types",
    "[khr_split_headers][builtins_integer]") {
  using abs_t = decltype(sycl::abs(std::declval<unsigned int>()));
  using max_t = decltype(sycl::max(std::declval<unsigned long>(),
                                   std::declval<unsigned long>()));
  STATIC_REQUIRE(std::is_same_v<abs_t, unsigned int>);
  STATIC_REQUIRE(std::is_same_v<max_t, unsigned long>);
}

}  // namespace khr_split_headers::tests
