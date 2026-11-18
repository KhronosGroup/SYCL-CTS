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
#include <sycl/khr/split_headers/builtins_math.hpp>
#include <sycl/khr/split_headers/multi_ptr.hpp>
#include <type_traits>
#include <utility>

// These tests verify that the builtin math header composes with
// <sycl/khr/split_headers/multi_ptr.hpp>. The math functions that write an
// output through a pointer (frexp, modf, sincos, fract, lgamma_r, remquo)
// accept either a raw C++ pointer or a multi_ptr whose Space is one of the
// writeable address spaces. Here we check the multi_ptr form for a scalar
// input; the return type is the scalar floating point type.

namespace khr_split_headers::tests {

// A multi_ptr to T in the global address space (a writeable address space).
template <typename T>
using global_ptr = sycl::multi_ptr<T, sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>;

TEST_CASE("builtins compose with multi_ptr: frexp",
          "[khr_split_headers][builtins_multi_ptr]") {
  using return_t = decltype(sycl::frexp(std::declval<float>(),
                                        std::declval<global_ptr<int>>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("builtins compose with multi_ptr: modf",
          "[khr_split_headers][builtins_multi_ptr]") {
  using return_t = decltype(sycl::modf(std::declval<float>(),
                                       std::declval<global_ptr<float>>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("builtins compose with multi_ptr: sincos",
          "[khr_split_headers][builtins_multi_ptr]") {
  using return_t = decltype(sycl::sincos(std::declval<float>(),
                                         std::declval<global_ptr<float>>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("builtins compose with multi_ptr: fract",
          "[khr_split_headers][builtins_multi_ptr]") {
  using return_t = decltype(sycl::fract(std::declval<float>(),
                                        std::declval<global_ptr<float>>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("builtins compose with multi_ptr: lgamma_r",
          "[khr_split_headers][builtins_multi_ptr]") {
  using return_t = decltype(sycl::lgamma_r(std::declval<float>(),
                                           std::declval<global_ptr<int>>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("builtins compose with multi_ptr: remquo",
          "[khr_split_headers][builtins_multi_ptr]") {
  using return_t =
      decltype(sycl::remquo(std::declval<float>(), std::declval<float>(),
                            std::declval<global_ptr<int>>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("builtins compose with multi_ptr: local address space is writeable",
          "[khr_split_headers][builtins_multi_ptr]") {
  using local_ptr =
      sycl::multi_ptr<int, sycl::access::address_space::local_space,
                      sycl::access::decorated::no>;
  using return_t =
      decltype(sycl::frexp(std::declval<float>(), std::declval<local_ptr>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the builtins_multi_ptr tests require sycl_khr_split_headers",
          "[khr_split_headers][builtins_multi_ptr]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
