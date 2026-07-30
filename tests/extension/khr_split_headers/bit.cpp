/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/bit.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the bit header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][bit]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the bit header defines the bit_cast function template",
          "[khr_split_headers][bit]") {
  using return_t = decltype(sycl::bit_cast<float>(0));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

}  // namespace khr_split_headers::tests
