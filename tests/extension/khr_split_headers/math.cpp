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
#include <sycl/khr/split_headers/math.hpp>
#include <type_traits>
#include <utility>

// <sycl/khr/split_headers/math.hpp> is an umbrella header that includes every
// SYCL built-in function family: math functions, native-precision math
// functions, half-precision math functions, integer functions, common
// functions, geometric functions and relational functions. These tests verify
// that a single include of math.hpp makes one representative function from each
// of those families available, using scalar inputs so the header can be
// validated standalone.

namespace khr_split_headers::tests {

TEST_CASE("the math header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][math]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the math header provides math functions",
          "[khr_split_headers][math]") {
  using sin_t = decltype(sycl::sin(std::declval<float>()));
  using pow_t =
      decltype(sycl::pow(std::declval<float>(), std::declval<float>()));
  using fma_t = decltype(sycl::fma(std::declval<float>(), std::declval<float>(),
                                   std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<sin_t, float>);
  STATIC_REQUIRE(std::is_same_v<pow_t, float>);
  STATIC_REQUIRE(std::is_same_v<fma_t, float>);
}

TEST_CASE("the math header provides native-precision math functions",
          "[khr_split_headers][math]") {
  using cos_t = decltype(sycl::native::cos(std::declval<float>()));
  using divide_t = decltype(sycl::native::divide(std::declval<float>(),
                                                 std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<cos_t, float>);
  STATIC_REQUIRE(std::is_same_v<divide_t, float>);
}

TEST_CASE("the math header provides half-precision math functions",
          "[khr_split_headers][math]") {
  using sqrt_t = decltype(sycl::half_precision::sqrt(std::declval<float>()));
  using log_t = decltype(sycl::half_precision::log(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<sqrt_t, float>);
  STATIC_REQUIRE(std::is_same_v<log_t, float>);
}

TEST_CASE("the math header provides integer functions",
          "[khr_split_headers][math]") {
  using abs_t = decltype(sycl::abs(std::declval<int>()));
  using clz_t = decltype(sycl::clz(std::declval<int>()));
  using mul24_t = decltype(sycl::mul24(std::declval<std::int32_t>(),
                                       std::declval<std::int32_t>()));
  STATIC_REQUIRE(std::is_same_v<abs_t, int>);
  STATIC_REQUIRE(std::is_same_v<clz_t, int>);
  STATIC_REQUIRE(std::is_same_v<mul24_t, std::int32_t>);
}

TEST_CASE("the math header provides common functions",
          "[khr_split_headers][math]") {
  using clamp_t = decltype(sycl::clamp(
      std::declval<float>(), std::declval<float>(), std::declval<float>()));
  using mix_t = decltype(sycl::mix(std::declval<float>(), std::declval<float>(),
                                   std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<clamp_t, float>);
  STATIC_REQUIRE(std::is_same_v<mix_t, float>);
}

TEST_CASE("the math header provides geometric functions",
          "[khr_split_headers][math]") {
  using dot_t =
      decltype(sycl::dot(std::declval<float>(), std::declval<float>()));
  using length_t = decltype(sycl::length(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<dot_t, float>);
  STATIC_REQUIRE(std::is_same_v<length_t, float>);
}

TEST_CASE("the math header provides relational functions",
          "[khr_split_headers][math]") {
  using isnan_t = decltype(sycl::isnan(std::declval<float>()));
  using isequal_t =
      decltype(sycl::isequal(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<isnan_t, bool>);
  STATIC_REQUIRE(std::is_same_v<isequal_t, bool>);
}

}  // namespace khr_split_headers::tests
