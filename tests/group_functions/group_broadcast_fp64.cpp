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

static auto queue = sycl_cts::util::get_cts_object::queue();

TEMPLATE_TEST_CASE_SIG("Group broadcast", "[group_func][fp64][dim]",
                       ((int D), D), 1, 2, 3) {
  // check dimension to only print warning once
  if constexpr (D == 1) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp fails to compile with segfault in the compiler. "
        "Skipping the test.");
#endif
  }

  // FIXME: clang-8: error: unable to execute command: Segmentation fault (core dumped)
  //        clang-8: error: spirv-ll-tool command failed due to signal (use -v to see invocation)
  //        Codeplay ComputeCpp - CE 2.11.0 Device Compiler - clang version 8.0.0  (based on LLVM 8.0.0svn)
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp64)) {
    broadcast_group<D, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
#endif
}

TEMPLATE_TEST_CASE_SIG("Sub-group broadcast and select",
                       "[group_func][fp64][dim]", ((int D), D), 1, 2, 3) {
  // check dimension to only print warning once
  if constexpr (D == 1) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp fails to compile with segfault in the compiler. "
        "Skipping the test.");
#endif
  }

  // FIXME: clang-8: error: unable to execute command: Segmentation fault (core dumped)
  //        clang-8: error: spirv-ll-tool command failed due to signal (use -v to see invocation)
  //        Codeplay ComputeCpp - CE 2.11.0 Device Compiler - clang version 8.0.0  (based on LLVM 8.0.0svn)
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp64)) {
    broadcast_sub_group<D, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
#endif
}
