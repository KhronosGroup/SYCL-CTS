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
#include <sycl/khr/split_headers/reduction.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the reduction header defines the initialize_to_identity class",
          "[khr_split_headers][reduction]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::property::reduction::initialize_to_identity>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the reduction tests require sycl_khr_split_headers",
          "[khr_split_headers][reduction]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
