/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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

#include "group_broadcast.h"

// FIXME: DPCPP does not implement group_broadcast for sycl::vec
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP)
using BroadcastTypes =
    concatenation<FundamentalTypes, std::tuple<bool, sycl::marray<float, 5>,
                                               sycl::marray<short int, 7>,
                                               util::custom_type>>::type;
// FIXME: ComputeCpp does not implement group_broadcast for sycl::vec,
//        sycl::marray, unsigned long long int, and long long int
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
using BroadcastTypes =
    std::tuple<size_t, float, char, signed char, unsigned char, short int,
               unsigned short int, int, unsigned int, long int,
               unsigned long int, bool, util::custom_type>;
#else
using BroadcastTypes = std::tuple<float, char, int, bool, util::custom_type>;
#endif
#else
using BroadcastTypes = CustomTypes;
#endif

static auto queue = sycl_cts::util::get_cts_object::queue();

TEMPLATE_LIST_TEST_CASE("Group broadcast", "[group_func][type_list][dim]",
                        BroadcastTypes) {
  // check type to only print warning once
  if constexpr (std::is_same_v<TestType, char>) {
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    WARN(
        "DPCPP does not implement group_broadcast for vec types. "
        "Skipping those test cases.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp does not implement group_broadcast for vec, marray, "
        "unsigned long long int, and long long int types. Skipping those test "
        "cases.");
    WARN(
        "ComputeCpp fails to compile with segfault in the compiler. "
        "Skipping the test.");
#endif
  }

  // FIXME: Codeplay ComputeCpp - CE 2.11.0
  //        Device Compiler - clang version 8.0.0  (based on LLVM 8.0.0svn)
  //        clang-8: error: unable to execute command: Segmentation fault
  //        clang-8: error: spirv-ll-tool command failed due to signal
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  // check all work group dimensions
  broadcast_group<1, TestType>(queue);
  broadcast_group<2, TestType>(queue);
  broadcast_group<3, TestType>(queue);
#endif
}

TEMPLATE_LIST_TEST_CASE("Sub-group broadcast and select",
                        "[group_func][type_list][dim]", BroadcastTypes) {
  // check type to only print warning once
  if constexpr (std::is_same_v<TestType, char>) {
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    WARN(
        "DPCPP does not implement group_broadcast for vec types. "
        "Skipping those test cases.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp does not implement group_broadcast for vec, marray, "
        "unsigned long long int, and long long int types. Skipping those test "
        "cases.");
    WARN(
        "ComputeCpp fails to compile with segfault in the compiler. "
        "Skipping the test.");
#endif
  }

  // FIXME: Codeplay ComputeCpp - CE 2.11.0
  //        Device Compiler - clang version 8.0.0  (based on LLVM 8.0.0svn)
  //        clang-8: error: unable to execute command: Segmentation fault
  //        clang-8: error: spirv-ll-tool command failed due to signal
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  // check all work group dimensions
  broadcast_sub_group<1, TestType>(queue);
  broadcast_sub_group<2, TestType>(queue);
  broadcast_sub_group<3, TestType>(queue);
#endif
}
