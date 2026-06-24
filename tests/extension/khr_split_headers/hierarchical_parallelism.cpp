/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2026 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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
