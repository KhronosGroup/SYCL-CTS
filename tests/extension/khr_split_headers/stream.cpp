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
#include <sycl/khr/split_headers/stream.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the stream header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][stream]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the stream header defines the stream_manipulator enum",
          "[khr_split_headers][stream]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::stream_manipulator>);
}

TEST_CASE("the stream header defines the setprecision function",
          "[khr_split_headers][stream]") {
  using return_t = decltype(sycl::setprecision(0));
  STATIC_REQUIRE(!std::is_same_v<return_t, void>);
}

TEST_CASE("the stream header defines the setw function",
          "[khr_split_headers][stream]") {
  using return_t = decltype(sycl::setw(0));
  STATIC_REQUIRE(!std::is_same_v<return_t, void>);
}

TEST_CASE("the stream header defines the stream class",
          "[khr_split_headers][stream]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::stream>);
}

TEST_CASE("the stream header defines the stream insertion operator template",
          "[khr_split_headers][stream]") {
  using return_t = decltype(std::declval<const sycl::stream&>()
                            << std::declval<const int&>());
  STATIC_REQUIRE(std::is_same_v<return_t, const sycl::stream&>);
}

}  // namespace khr_split_headers::tests
