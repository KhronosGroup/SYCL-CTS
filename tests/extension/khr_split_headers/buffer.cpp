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
