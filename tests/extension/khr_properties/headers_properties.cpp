/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/properties.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/properties.hpp>

#include <type_traits>

namespace khr_properties::tests {

TEST_CASE(
    "the properties header defines the properties class and the property "
    "traits",
    "[khr_properties][headers]") {
  STATIC_CHECK(std::is_same_v<sycl::khr::empty_properties_t,
                              decltype(sycl::khr::properties{})>);
  STATIC_CHECK_FALSE(sycl::khr::is_property_v<int>);
  STATIC_CHECK_FALSE(sycl::khr::is_property_key_v<int>);
  STATIC_CHECK_FALSE(sycl::khr::is_property_key_compile_time_v<int>);
  STATIC_CHECK_FALSE(sycl::khr::is_property_for_v<int, int>);
  STATIC_CHECK_FALSE(sycl::khr::is_property_key_for_v<int, int>);
  STATIC_CHECK(
      sycl::khr::is_property_list_for_v<sycl::khr::empty_properties_t, int>);
  STATIC_CHECK(std::is_base_of_v<std::false_type, sycl::khr::is_property<int>>);
  STATIC_CHECK(
      std::is_base_of_v<std::false_type, sycl::khr::is_property_key<int>>);
  STATIC_CHECK(std::is_base_of_v<std::false_type,
                                 sycl::khr::is_property_key_compile_time<int>>);
  STATIC_CHECK(
      std::is_base_of_v<std::false_type, sycl::khr::is_property_for<int, int>>);
  STATIC_CHECK(std::is_base_of_v<std::false_type,
                                 sycl::khr::is_property_key_for<int, int>>);
  STATIC_CHECK(
      std::is_base_of_v<
          std::true_type,
          sycl::khr::is_property_list_for<sycl::khr::empty_properties_t, int>>);
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
