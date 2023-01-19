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

// need also double tests enabled
#ifdef SYCL_CTS_ENABLE_DOUBLE_TESTS

#include "group_reduce.h"

TEMPLATE_TEST_CASE_SIG("Group and sub-group joint reduce functions with init",
                       "[group_func][fp16][fp64][dim]", ((int D), D), 1, 2, 3) {
  auto queue = sycl_cts::util::get_cts_object::queue();

  // check dimensions to only print warning once
  if constexpr (D == 1) {
    // FIXME: hipSYCL omission
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
    WARN(
        "hipSYCL has no implementation of T joint_reduce(sub_group g, Ptr "
        "first, Ptr last, T init, "
        "BinaryOperation binary_op) over sub-groups. Skipping the test case.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp cannot handle cases of different types. "
        "Skipping such test cases.");
    WARN("ComputeCpp cannot handle half type. Skipping the test.");
#elif defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    WARN(
        "DPCPP cannot handle cases of different types. "
        "Skipping such test cases.");
#endif
  }

  // FIXME: ComputeCpp has no half
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
  // FIXME: DPCPP and ComputeCpp cannot handle cases of different types
#elif defined(SYCL_CTS_COMPILING_WITH_DPCPP) || \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    init_joint_reduce_group<D, sycl::half, double>(queue);
    init_joint_reduce_group<D, double, sycl::half>(queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
}

TEMPLATE_TEST_CASE_SIG("Group and sub-group reduce functions with init",
                       "[group_func][fp16][fp64][dim]", ((int D), D), 1, 2, 3) {
  auto queue = sycl_cts::util::get_cts_object::queue();

  // check dimensions to only print warning once
  if constexpr (D == 1) {
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    WARN(
        "DPCPP cannot handle cases of different types. "
        "Skipping such test cases.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp cannot handle cases of different types. "
        "Skipping such test cases.");
    WARN("ComputeCpp cannot handle half type. Skipping the test.");
#endif
  }

  // FIXME: ComputeCpp has no half
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
  // FIXME: DPCPP and ComputeCpp cannot handle cases of different types
#elif defined(SYCL_CTS_COMPILING_WITH_DPCPP) || \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    init_reduce_over_group<D, sycl::half, double>(queue);
    init_reduce_over_group<D, double, sycl::half>(queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
}

#endif
