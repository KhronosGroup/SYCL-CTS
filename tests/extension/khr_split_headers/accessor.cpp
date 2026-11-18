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
#include <sycl/khr/split_headers/accessor.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the accessor header defines the accessor class template",
          "[khr_split_headers][accessor]") {
  using accessor_t = sycl::accessor<int, 1, sycl::access_mode::read_write>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<accessor_t>);
}

TEST_CASE("the accessor header defines the host accessor class template",
          "[khr_split_headers][accessor]") {
  using accessor_t = sycl::host_accessor<int, 1, sycl::access_mode::read_write>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<accessor_t>);
}

TEST_CASE("the accessor header defines the local_accessor class template",
          "[khr_split_headers][accessor]") {
  using accessor_t = sycl::local_accessor<int, 1>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<accessor_t>);
}

}  // namespace khr_split_headers::tests

#else

// The extension is not implemented, so none of the headers under test exist.
// Keep one always-present test case to report the category as skipped.
TEST_CASE("the accessor tests require sycl_khr_split_headers",
          "[khr_split_headers][accessor]") {
  SKIP("SYCL_KHR_SPLIT_HEADERS is not defined");
}

#endif  // SYCL_KHR_SPLIT_HEADERS
