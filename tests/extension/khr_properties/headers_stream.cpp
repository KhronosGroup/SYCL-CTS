/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/split_headers/stream.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/properties.hpp>
#include <sycl/khr/split_headers/stream.hpp>

#include <cstddef>
#include <type_traits>

namespace khr_properties::tests {

TEST_CASE("the stream header defines the new stream constructor",
          "[khr_properties][headers]") {
  STATIC_CHECK(
      std::is_constructible_v<sycl::stream, std::size_t, std::size_t,
                              sycl::handler&, sycl::khr::empty_properties_t>);
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
