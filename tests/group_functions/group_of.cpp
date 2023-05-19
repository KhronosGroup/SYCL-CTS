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

#include "group_of.h"

// use wide types to exclude truncation of init values
using WideTypes = std::tuple<int32_t, uint32_t, int64_t, uint64_t, float>;

TEMPLATE_LIST_TEST_CASE("Group and sub_group joint of bool functions",
                        "[group_func][type_list][dim]", WideTypes) {
  auto queue = once_per_unit::get_queue();
  // check type to only print warning once
  if constexpr (std::is_same_v<TestType, float>) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
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
  joint_of_group<1, TestType>(queue);
  joint_of_group<2, TestType>(queue);
  joint_of_group<3, TestType>(queue);
#endif
}

TEMPLATE_LIST_TEST_CASE(
    "Group and sub_group of bool functions with predicate functions",
    "[group_func][type_list][dim]", WideTypes) {
  auto queue = once_per_unit::get_queue();
  // check type to only print warning once
  if constexpr (std::is_same_v<TestType, float>) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
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
  predicate_function_of_group<1, TestType>(queue);
  predicate_function_of_group<2, TestType>(queue);
  predicate_function_of_group<3, TestType>(queue);

  predicate_function_of_sub_group<1, TestType>(queue);
  predicate_function_of_sub_group<2, TestType>(queue);
  predicate_function_of_sub_group<3, TestType>(queue);
#endif
}

TEMPLATE_TEST_CASE_SIG("Group and sub_group of bool functions",
                       "[group_func][dim]", ((int D), D), 1, 2, 3) {
  auto queue = once_per_unit::get_queue();
  // check dimension to only print warning once
  if constexpr (D == 1) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
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
  bool_function_of_group<D>(queue);
  bool_function_of_sub_group<D>(queue);
#endif
}
