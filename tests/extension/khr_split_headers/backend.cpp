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
#include <sycl/khr/split_headers/backend.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the backend header defines the backend enum",
          "[khr_split_headers][backend]") {
  STATIC_REQUIRE(std::is_enum_v<sycl::backend>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the backend tests require sycl_khr_split_headers",
          "[khr_split_headers][backend]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
