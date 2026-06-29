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
#include <sycl/khr/split_headers/builtins_geometric.hpp>
#include <sycl/khr/split_headers/builtins_integer.hpp>
#include <sycl/khr/split_headers/builtins_math.hpp>
#include <sycl/khr/split_headers/builtins_relational.hpp>
#include <sycl/khr/split_headers/marray.hpp>
#include <cstdint>
#include <type_traits>
#include <utility>

// These tests verify that the builtin function headers compose with
// <sycl/khr/split_headers/marray.hpp>: when a builtin header is included
// together with the marray header, the marray overloads of the builtin
// functions become available. We check one representative function per family
// and verify the documented marray return types. Unlike vec, marray relational
// functions return marray<bool, N>.

namespace khr_split_headers::tests {

template <typename T, std::size_t N>
using m = sycl::marray<T, N>;

TEST_CASE("builtins compose with marray: integer functions",
          "[khr_split_headers][builtins_marray]") {
  using abs_t = decltype(sycl::abs(std::declval<m<int, 4>>()));
  using max_t =
      decltype(sycl::max(std::declval<m<int, 4>>(), std::declval<m<int, 4>>()));
  using clz_t = decltype(sycl::clz(std::declval<m<int, 4>>()));
  STATIC_REQUIRE(std::is_same_v<abs_t, m<int, 4>>);
  STATIC_REQUIRE(std::is_same_v<max_t, m<int, 4>>);
  STATIC_REQUIRE(std::is_same_v<clz_t, m<int, 4>>);
}

TEST_CASE("builtins compose with marray: upsample widens element type",
          "[khr_split_headers][builtins_marray]") {
  using up_t = decltype(sycl::upsample(std::declval<m<std::uint8_t, 4>>(),
                                       std::declval<m<std::uint8_t, 4>>()));
  STATIC_REQUIRE(std::is_same_v<up_t, m<std::uint16_t, 4>>);
}

TEST_CASE("builtins compose with marray: common functions",
          "[khr_split_headers][builtins_marray]") {
  using clamp_t =
      decltype(sycl::clamp(std::declval<m<float, 4>>(),
                           std::declval<m<float, 4>>(),
                           std::declval<m<float, 4>>()));
  using mix_t =
      decltype(sycl::mix(std::declval<m<float, 4>>(),
                         std::declval<m<float, 4>>(),
                         std::declval<m<float, 4>>()));
  STATIC_REQUIRE(std::is_same_v<clamp_t, m<float, 4>>);
  STATIC_REQUIRE(std::is_same_v<mix_t, m<float, 4>>);
}

TEST_CASE("builtins compose with marray: math functions",
          "[khr_split_headers][builtins_marray]") {
  using sin_t = decltype(sycl::sin(std::declval<m<float, 4>>()));
  using pow_t = decltype(sycl::pow(std::declval<m<float, 4>>(),
                                   std::declval<m<float, 4>>()));
  STATIC_REQUIRE(std::is_same_v<sin_t, m<float, 4>>);
  STATIC_REQUIRE(std::is_same_v<pow_t, m<float, 4>>);
}

TEST_CASE("builtins compose with marray: math function with pointer output",
          "[khr_split_headers][builtins_marray]") {
  using frexp_t = decltype(sycl::frexp(std::declval<m<float, 4>>(),
                                       std::declval<m<int, 4>*>()));
  STATIC_REQUIRE(std::is_same_v<frexp_t, m<float, 4>>);
}

TEST_CASE("builtins compose with marray: relational functions return bool marray",
          "[khr_split_headers][builtins_marray]") {
  using iseq_t = decltype(sycl::isequal(std::declval<m<float, 4>>(),
                                        std::declval<m<float, 4>>()));
  using isnan_t = decltype(sycl::isnan(std::declval<m<float, 4>>()));
  STATIC_REQUIRE(std::is_same_v<iseq_t, m<bool, 4>>);
  STATIC_REQUIRE(std::is_same_v<isnan_t, m<bool, 4>>);
}

TEST_CASE("builtins compose with marray: geometric functions",
          "[khr_split_headers][builtins_marray]") {
  using dot_t = decltype(sycl::dot(std::declval<m<float, 4>>(),
                                   std::declval<m<float, 4>>()));
  using cross_t = decltype(sycl::cross(std::declval<m<float, 3>>(),
                                       std::declval<m<float, 3>>()));
  using len_t = decltype(sycl::length(std::declval<m<float, 4>>()));
  // dot and length reduce an marray to its element type.
  STATIC_REQUIRE(std::is_same_v<dot_t, float>);
  STATIC_REQUIRE(std::is_same_v<len_t, float>);
  // cross is only defined for 3- and 4-element marrays.
  STATIC_REQUIRE(std::is_same_v<cross_t, m<float, 3>>);
}

}  // namespace khr_split_headers::tests
