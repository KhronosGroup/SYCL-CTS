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
#include <sycl/khr/split_headers/byte.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the byte header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][byte]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the byte header defines the byte class",
          "[khr_split_headers][byte]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::byte>);
}

}  // namespace khr_split_headers::tests
