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
#include <sycl/khr/split_headers/groups.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the groups header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][groups]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the groups header defines the group class template",
          "[khr_split_headers][groups]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::group<1>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::group<2>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::group<3>>);
}

TEST_CASE("the groups header defines the sub_group class",
          "[khr_split_headers][groups]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::sub_group>);
}

TEST_CASE("the groups header defines the device_event class",
          "[khr_split_headers][groups]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::device_event>);
}

}  // namespace khr_split_headers::tests
