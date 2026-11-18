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

#if SYCL_CTS_KHR_SPLIT_HEADERS
#include <sycl/khr/split_headers/kernel_bundle.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the kernel_bundle header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][kernel_bundle]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the kernel_bundle header defines the kernel_bundle class template",
          "[khr_split_headers][kernel_bundle]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::kernel_bundle<sycl::bundle_state::input>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::kernel_bundle<sycl::bundle_state::object>>);
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::kernel_bundle<sycl::bundle_state::executable>>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the kernel_bundle tests require sycl_khr_split_headers",
          "[khr_split_headers][kernel_bundle]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_CTS_KHR_SPLIT_HEADERS
