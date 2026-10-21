/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/split_headers/context.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/properties.hpp>
#include <sycl/khr/split_headers/context.hpp>

#include <type_traits>
#include <vector>

namespace khr_properties::tests {

TEST_CASE("the context header defines the new context constructors",
          "[khr_properties][headers]") {
  using sycl::context;
  using e_t = sycl::khr::empty_properties_t;
  using ah_t = sycl::async_handler;
  using dev_t = const sycl::device&;
  using plt_t = const sycl::platform&;
  using devs_t = const std::vector<sycl::device>&;
  STATIC_CHECK(std::is_constructible_v<context, e_t>);
  STATIC_CHECK(std::is_constructible_v<context, ah_t, e_t>);
  STATIC_CHECK(std::is_constructible_v<context, dev_t, e_t>);
  STATIC_CHECK(std::is_constructible_v<context, dev_t, ah_t, e_t>);
  STATIC_CHECK(std::is_constructible_v<context, plt_t, e_t>);
  STATIC_CHECK(std::is_constructible_v<context, plt_t, ah_t, e_t>);
  STATIC_CHECK(std::is_constructible_v<context, devs_t, e_t>);
  STATIC_CHECK(std::is_constructible_v<context, devs_t, ah_t, e_t>);
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
