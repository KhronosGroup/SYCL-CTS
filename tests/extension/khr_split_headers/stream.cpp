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
#include <sycl/khr/split_headers/stream.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

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

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the stream tests require sycl_khr_split_headers",
          "[khr_split_headers][stream]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
