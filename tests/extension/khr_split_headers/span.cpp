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
#include <sycl/khr/split_headers/span.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the span header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][span]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the span header defines the span class template",
          "[khr_split_headers][span]") {
  using span_t = sycl::span<int, 1>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<span_t>);
}

TEST_CASE("the span header defines the dynamic_extent constant",
          "[khr_split_headers][span]") {
  STATIC_REQUIRE(
      std::is_same_v<decltype(sycl::dynamic_extent), const std::size_t>);
}

}  // namespace khr_split_headers::tests
