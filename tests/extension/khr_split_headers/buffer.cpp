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

#if SYCL_CTS_KHR_SPLIT_HEADERS
#include <sycl/khr/split_headers/buffer.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the buffer header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][buffer]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the buffer header defines the buffer class template",
          "[khr_split_headers][buffer]") {
  using buffer_t = sycl::buffer<int, 1>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<buffer_t>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the buffer tests require sycl_khr_split_headers",
          "[khr_split_headers][buffer]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_CTS_KHR_SPLIT_HEADERS
