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
#include <sycl/khr/split_headers/builtins_common.hpp>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/builtins_common.hpp>
// provides the common functions from the SYCL specification. We use scalar
// "generic floating point types" (plain float/double) so that the header can be
// validated standalone, without pulling in vec/marray headers. Each function's
// return type equals its (first) floating point argument type for scalar
// inputs.

namespace khr_split_headers::tests {

TEST_CASE("the builtins_common header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][builtins_common]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the builtins_common header defines the clamp function",
          "[khr_split_headers][builtins_common]") {
  using return_t = decltype(sycl::clamp(
      std::declval<float>(), std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the degrees function",
          "[khr_split_headers][builtins_common]") {
  using return_t = decltype(sycl::degrees(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the max function",
          "[khr_split_headers][builtins_common]") {
  using return_t =
      decltype(sycl::max(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the min function",
          "[khr_split_headers][builtins_common]") {
  using return_t =
      decltype(sycl::min(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the mix function",
          "[khr_split_headers][builtins_common]") {
  using return_t = decltype(sycl::mix(
      std::declval<float>(), std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the radians function",
          "[khr_split_headers][builtins_common]") {
  using return_t = decltype(sycl::radians(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the step function",
          "[khr_split_headers][builtins_common]") {
  using return_t =
      decltype(sycl::step(std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the smoothstep function",
          "[khr_split_headers][builtins_common]") {
  using return_t = decltype(sycl::smoothstep(
      std::declval<float>(), std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header defines the sign function",
          "[khr_split_headers][builtins_common]") {
  using return_t = decltype(sycl::sign(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_common header provides functions for double scalars",
          "[khr_split_headers][builtins_common]") {
  using clamp_t = decltype(sycl::clamp(
      std::declval<double>(), std::declval<double>(), std::declval<double>()));
  using sign_t = decltype(sycl::sign(std::declval<double>()));
  STATIC_REQUIRE(std::is_same_v<clamp_t, double>);
  STATIC_REQUIRE(std::is_same_v<sign_t, double>);
}

}  // namespace khr_split_headers::tests
