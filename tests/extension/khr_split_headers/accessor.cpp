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
#include <sycl/khr/split_headers/accessor.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the accessor header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][accessor]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

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
