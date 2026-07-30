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
#include <sycl/khr/split_headers/hierarchical_parallelism.hpp>

namespace khr_split_headers::tests {

TEST_CASE(
    "the hierarchical_parallelism header defines the SYCL_KHR_SPLIT_HEADERS "
    "macro",
    "[khr_split_headers][hierarchical_parallelism]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE(
    "the hierarchical_parallelism header defines the private_memory class "
    "template",
    "[khr_split_headers][hierarchical_parallelism]") {
  using private_memory_t = sycl::private_memory<int, 1>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<private_memory_t>);
}

TEST_CASE(
    "the hierarchical_parallelism header defines the h_item class template",
    "[khr_split_headers][hierarchical_parallelism]") {
  using h_item_t = sycl::h_item<1>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<h_item_t>);
}

}  // namespace khr_split_headers::tests
