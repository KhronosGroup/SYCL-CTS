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
#include <sycl/khr/split_headers/vec.hpp>
#include <cstdint>
#include <type_traits>
#include <utility>

// These tests verify that the builtin function headers compose with
// <sycl/khr/split_headers/vec.hpp>: when a builtin header is included together
// with the vec header, the vec overloads of the builtin functions become
// available. We check one representative function per family and verify the
// documented vec return types.

namespace khr_split_headers::tests {

template <typename T, int N>
using v = sycl::vec<T, N>;

TEST_CASE("builtins compose with vec: integer functions",
          "[khr_split_headers][builtins_vec]") {
  using abs_t = decltype(sycl::abs(std::declval<v<int, 4>>()));
  using max_t =
      decltype(sycl::max(std::declval<v<int, 4>>(), std::declval<v<int, 4>>()));
  using clz_t = decltype(sycl::clz(std::declval<v<int, 4>>()));
  STATIC_REQUIRE(std::is_same_v<abs_t, v<int, 4>>);
  STATIC_REQUIRE(std::is_same_v<max_t, v<int, 4>>);
  STATIC_REQUIRE(std::is_same_v<clz_t, v<int, 4>>);
}

TEST_CASE("builtins compose with vec: upsample widens element type",
          "[khr_split_headers][builtins_vec]") {
  using up_t = decltype(sycl::upsample(std::declval<v<std::uint8_t, 4>>(),
                                       std::declval<v<std::uint8_t, 4>>()));
  STATIC_REQUIRE(std::is_same_v<up_t, v<std::uint16_t, 4>>);
}

TEST_CASE("builtins compose with vec: common functions",
          "[khr_split_headers][builtins_vec]") {
  using clamp_t =
      decltype(sycl::clamp(std::declval<v<float, 4>>(),
                           std::declval<v<float, 4>>(),
                           std::declval<v<float, 4>>()));
  using mix_t =
      decltype(sycl::mix(std::declval<v<float, 4>>(),
                         std::declval<v<float, 4>>(),
                         std::declval<v<float, 4>>()));
  STATIC_REQUIRE(std::is_same_v<clamp_t, v<float, 4>>);
  STATIC_REQUIRE(std::is_same_v<mix_t, v<float, 4>>);
}

TEST_CASE("builtins compose with vec: math functions",
          "[khr_split_headers][builtins_vec]") {
  using sin_t = decltype(sycl::sin(std::declval<v<float, 4>>()));
  using pow_t = decltype(sycl::pow(std::declval<v<float, 4>>(),
                                   std::declval<v<float, 4>>()));
  STATIC_REQUIRE(std::is_same_v<sin_t, v<float, 4>>);
  STATIC_REQUIRE(std::is_same_v<pow_t, v<float, 4>>);
}

TEST_CASE("builtins compose with vec: math function with pointer output",
          "[khr_split_headers][builtins_vec]") {
  using frexp_t = decltype(sycl::frexp(std::declval<v<float, 4>>(),
                                       std::declval<v<std::int32_t, 4>*>()));
  STATIC_REQUIRE(std::is_same_v<frexp_t, v<float, 4>>);
}

TEST_CASE("builtins compose with vec: relational functions return integer vec",
          "[khr_split_headers][builtins_vec]") {
  using iseq_t = decltype(sycl::isequal(std::declval<v<float, 4>>(),
                                        std::declval<v<float, 4>>()));
  using isnan_t = decltype(sycl::isnan(std::declval<v<float, 4>>()));
  STATIC_REQUIRE(std::is_same_v<iseq_t, v<std::int32_t, 4>>);
  STATIC_REQUIRE(std::is_same_v<isnan_t, v<std::int32_t, 4>>);
}

TEST_CASE("builtins compose with vec: geometric functions",
          "[khr_split_headers][builtins_vec]") {
  using dot_t = decltype(sycl::dot(std::declval<v<float, 4>>(),
                                   std::declval<v<float, 4>>()));
  using cross_t = decltype(sycl::cross(std::declval<v<float, 3>>(),
                                       std::declval<v<float, 3>>()));
  using len_t = decltype(sycl::length(std::declval<v<float, 4>>()));
  // dot and length reduce a vec to its element type.
  STATIC_REQUIRE(std::is_same_v<dot_t, float>);
  STATIC_REQUIRE(std::is_same_v<len_t, float>);
  // cross is only defined for 3- and 4-element vectors.
  STATIC_REQUIRE(std::is_same_v<cross_t, v<float, 3>>);
}

}  // namespace khr_split_headers::tests
