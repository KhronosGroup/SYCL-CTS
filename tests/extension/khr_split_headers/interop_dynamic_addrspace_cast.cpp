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

// The sycl_khr_dynamic_addrspace_cast extension states that when an
// implementation also implements sycl_khr_split_headers, the APIs defined in
// the sycl::khr namespace are additionally provided by the header
// <sycl/khr/dynamic_addrspace_cast.hpp>, and the feature test macro is
// additionally provided by <sycl/khr/split_headers/version.hpp>. These tests
// verify that coexistence and are guarded on both SYCL_KHR_SPLIT_HEADERS and
// SYCL_KHR_DYNAMIC_ADDRSPACE_CAST.

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/version.hpp>

#if defined(SYCL_KHR_SPLIT_HEADERS) && defined(SYCL_KHR_DYNAMIC_ADDRSPACE_CAST)

#include <sycl/khr/dynamic_addrspace_cast.hpp>
#include <sycl/khr/split_headers/multi_ptr.hpp>
#include <type_traits>
#include <utility>

namespace khr_split_headers::tests {

TEST_CASE(
    "the dynamic_addrspace_cast API is provided through "
    "<sycl/khr/dynamic_addrspace_cast.hpp> when split_headers is implemented",
    "[khr_split_headers][dynamic_addrspace_cast]") {
  // khr::dynamic_addrspace_cast takes a generic-space multi_ptr and produces a
  // multi_ptr in the requested address space.
  using generic_ptr_t =
      sycl::multi_ptr<int, sycl::access::address_space::generic_space,
                      sycl::access::decorated::no>;
  using return_t = decltype(sycl::khr::dynamic_addrspace_cast<
                            sycl::access::address_space::global_space>(
      std::declval<generic_ptr_t>()));
  using expected_t =
      sycl::multi_ptr<int, sycl::access::address_space::global_space,
                      sycl::access::decorated::no>;
  STATIC_REQUIRE(std::is_same_v<return_t, expected_t>);
}

TEST_CASE(
    "the dynamic_addrspace_cast feature test macro is provided through "
    "<sycl/khr/split_headers/version.hpp>",
    "[khr_split_headers][dynamic_addrspace_cast]") {
  STATIC_REQUIRE(SYCL_KHR_DYNAMIC_ADDRSPACE_CAST >= 1);
}

}  // namespace khr_split_headers::tests

#else

TEST_CASE("dynamic_addrspace_cast split_headers coexistence is not applicable",
          "[khr_split_headers][dynamic_addrspace_cast]") {
  SUCCEED(
      "SYCL_KHR_SPLIT_HEADERS and/or SYCL_KHR_DYNAMIC_ADDRSPACE_CAST not "
      "implemented");
}

#endif
