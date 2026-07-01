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

// The sycl_khr_group_interface extension states that when an implementation
// also implements sycl_khr_split_headers, the APIs defined in the sycl::khr
// namespace are additionally provided by the header
// <sycl/khr/group_interface.hpp>, and the feature test macro is additionally
// provided by <sycl/khr/split_headers/version.hpp>. These tests verify that
// coexistence. They are only meaningful when both extensions are implemented,
// so they are guarded on both SYCL_KHR_SPLIT_HEADERS and SYCL_KHR_GROUP_INTERFACE.

#include "util.h"
#include <catch2/catch_test_macros.hpp>
// version.hpp of the split_headers extension is required to expose the feature
// test macros of all other supported extensions.
#include <sycl/khr/split_headers/version.hpp>

#if defined(SYCL_KHR_SPLIT_HEADERS) && defined(SYCL_KHR_GROUP_INTERFACE)

#include <sycl/khr/group_interface.hpp>
#include <sycl/khr/split_headers/groups.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE(
    "the group_interface APIs are provided through <sycl/khr/group_interface.hpp>"
    " when split_headers is implemented",
    "[khr_split_headers][group_interface]") {
  // The sycl::khr::work_group group-interface wrapper is available.
  using wg_t = sycl::khr::work_group<1>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<wg_t>);
}

TEST_CASE(
    "the group_interface feature test macro is provided through "
    "<sycl/khr/split_headers/version.hpp>",
    "[khr_split_headers][group_interface]") {
  STATIC_REQUIRE(SYCL_KHR_GROUP_INTERFACE >= 1);
}

}  // namespace khr_split_headers::tests

#else

// When either extension is not implemented the coexistence requirement does not
// apply; keep at least one always-present test case so the translation unit is
// not empty.
TEST_CASE(
    "group_interface split_headers coexistence is not applicable",
    "[khr_split_headers][group_interface]") {
  SUCCEED(
      "SYCL_KHR_SPLIT_HEADERS and/or SYCL_KHR_GROUP_INTERFACE not implemented");
}

#endif
