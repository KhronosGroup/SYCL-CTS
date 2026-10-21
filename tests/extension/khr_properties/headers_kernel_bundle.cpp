/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/split_headers/kernel_bundle.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/properties.hpp>
#include <sycl/khr/split_headers/kernel_bundle.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace khr_properties::tests {

TEST_CASE(
    "the kernel_bundle header defines the compile, link and build tags and "
    "overloads",
    "[khr_properties][headers]") {
  using e_t = sycl::khr::empty_properties_t;
  using input_t = sycl::kernel_bundle<sycl::bundle_state::input>;
  using object_t = sycl::kernel_bundle<sycl::bundle_state::object>;
  using exe_t = sycl::kernel_bundle<sycl::bundle_state::executable>;
  using devs_t = const std::vector<sycl::device>&;

  STATIC_CHECK(
      sycl::khr::is_property_list_for_v<e_t,
                                        sycl::khr::property::tag::compile>);
  STATIC_CHECK(
      sycl::khr::is_property_list_for_v<e_t, sycl::khr::property::tag::link>);
  STATIC_CHECK(
      sycl::khr::is_property_list_for_v<e_t, sycl::khr::property::tag::build>);

  STATIC_CHECK(
      std::is_same_v<decltype(sycl::compile(std::declval<const input_t&>(),
                                            std::declval<devs_t>(), e_t{})),
                     object_t>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::compile(
                                  std::declval<const input_t&>(), e_t{})),
                              object_t>);
  STATIC_CHECK(
      std::is_same_v<
          decltype(sycl::link(std::declval<const object_t&>(), e_t{})), exe_t>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::link(
                                  std::declval<const std::vector<object_t>&>(),
                                  std::declval<devs_t>(), e_t{})),
                              exe_t>);
  STATIC_CHECK(
      std::is_same_v<
          decltype(sycl::build(std::declval<const input_t&>(), e_t{})), exe_t>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::build(std::declval<const input_t&>(),
                                          std::declval<devs_t>(), e_t{})),
                     exe_t>);
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
