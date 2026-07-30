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
#include <sycl/khr/split_headers/interop_handle.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the interhop_handle header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][interop_handle]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the interop_handle header defines the interop_handle class",
          "[khr_split_headers][interop_handle]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::interop_handle>);
}

}  // namespace khr_split_headers::tests
