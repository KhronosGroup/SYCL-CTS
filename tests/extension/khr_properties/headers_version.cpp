/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/split_headers/version.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/split_headers/version.hpp>

namespace khr_properties::tests {

TEST_CASE("the version header defines the SYCL_KHR_PROPERTIES macro",
          "[khr_properties][headers]") {
#ifndef SYCL_KHR_PROPERTIES
  static_assert(false, "SYCL_KHR_PROPERTIES is not defined");
#else
  STATIC_CHECK(SYCL_KHR_PROPERTIES == 1);
#endif
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
