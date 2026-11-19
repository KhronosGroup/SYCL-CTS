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
#include <sycl/khr/split_headers/property_list.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the property_list header defines the property_list class",
          "[khr_split_headers][property_list]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::property_list>);
}

TEST_CASE("the property_list header defines the is_property trait",
          "[khr_split_headers][property_list]") {
  // A type that is not a property yields a well-formed false value.
  STATIC_REQUIRE(
      std::is_same_v<decltype(sycl::is_property<int>::value), const bool>);
  STATIC_REQUIRE(sycl::is_property_v<int> == false);
}

TEST_CASE("the property_list header defines the is_property_of trait",
          "[khr_split_headers][property_list]") {
  STATIC_REQUIRE(std::is_same_v<decltype(sycl::is_property_of<int, int>::value),
                                const bool>);
  STATIC_REQUIRE(sycl::is_property_of_v<int, int> == false);
}

}  // namespace khr_split_headers::tests

#endif  // SYCL_KHR_SPLIT_HEADERS
