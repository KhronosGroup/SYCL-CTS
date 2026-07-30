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
#include <sycl/khr/split_headers/atomic.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the atomic header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][atomic]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

TEST_CASE("the atomic header defines the memory_order enum",
          "[khr_split_headers][atomic]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::memory_order>);
}

TEST_CASE("the atomic header defines the memory_scope enum",
          "[khr_split_headers][atomic]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::memory_scope>);
}

TEST_CASE("the atomic header defines the atomic_fence function",
          "[khr_split_headers][atomic]") {
  using return_t = decltype(sycl::atomic_fence(sycl::memory_order::seq_cst,
                                               sycl::memory_scope::device));
  STATIC_REQUIRE(std::is_void_v<return_t>);
}

TEST_CASE("the atomic header defines the atomic_ref class template",
          "[khr_split_headers][atomic]") {
  using atomic_ref_t = sycl::atomic_ref<int, sycl::memory_order::seq_cst,
                                        sycl::memory_scope::device>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<atomic_ref_t>);
}

TEST_CASE("the atomic header defines the atomic class template",
          "[khr_split_headers][atomic]") {
  using atomic_t = sycl::atomic<int>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<atomic_t>);
}

TEST_CASE("the atomic header defines the atomic_load function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_load(std::declval<sycl::atomic<int>>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the atomic header defines the atomic_store function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_store(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, void>);
}

TEST_CASE("the atomic header defines the atomic_exchange function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_exchange(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE(
    "the atomic header defines the atomic_compare_exchange_strong function "
    "template",
    "[khr_split_headers][atomic]") {
  using return_t = decltype(sycl::atomic_compare_exchange_strong(
      std::declval<sycl::atomic<int>>(), std::declval<int&>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the atomic header defines the atomic_fetch_add function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_fetch_add(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the atomic header defines the atomic_fetch_sub function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_fetch_sub(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the atomic header defines the atomic_fetch_and function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_fetch_and(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the atomic header defines the atomic_fetch_or function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_fetch_or(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the atomic header defines the atomic_fetch_xor function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_fetch_xor(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the atomic header defines the atomic_fetch_min function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_fetch_min(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the atomic header defines the atomic_fetch_max function template",
          "[khr_split_headers][atomic]") {
  using return_t =
      decltype(sycl::atomic_fetch_max(std::declval<sycl::atomic<int>>(), 0));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

}  // namespace khr_split_headers::tests
