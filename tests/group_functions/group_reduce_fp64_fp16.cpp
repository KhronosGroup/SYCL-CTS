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

// FIXME: ComputeCpp has no half
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
using ReduceTypes = unnamed_type_pack<>;
#else
using ReduceTypes = unnamed_type_pack<double, sycl::half>;
#endif

// 2-dim Cartesian product of type lists
using prod2 = product<std::tuple, ReduceTypes, ReduceTypes>::type;

TEMPLATE_LIST_TEST_CASE("Group and sub-group joint reduce functions with init",
                        "[group_func][fp16][fp64][dim]", prod2) {
  auto queue = once_per_unit::get_queue();
  using T = std::tuple_element_t<0, TestType>;
  using U = std::tuple_element_t<1, TestType>;

  // check types to only print warning once
  if constexpr (std::is_same_v<T, double> && std::is_same_v<U, double>) {
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
#endif
  }

  // FIXME: ComputeCpp has no half
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
  // FIXME: ComputeCpp cannot handle cases of different types
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    // Get binary operators from T
    const auto Operators = get_op_types<T>();
    const auto RetType = unnamed_type_pack<T>();
    const auto ReducedType = unnamed_type_pack<U>();
    // check all work group dimensions
    for_all_combinations<invoke_init_joint_reduce_group>(
        Dims, RetType, ReducedType, Operators, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
}

TEMPLATE_LIST_TEST_CASE("Group and sub-group reduce functions with init",
                        "[group_func][fp16][fp64][dim]", prod2) {
  auto queue = once_per_unit::get_queue();
  using T = std::tuple_element_t<0, TestType>;
  using U = std::tuple_element_t<1, TestType>;

  // check types to only print warning once
  if constexpr (std::is_same_v<T, double> && std::is_same_v<U, double>) {
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    // Link to issue https://github.com/intel/llvm/issues/8341
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
  // FIXME: ComputeCpp cannot handle cases of different types
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    // Get binary operators from T
    const auto Operators = get_op_types<T>();
    const auto RetType = unnamed_type_pack<T>();
    const auto ReducedType = unnamed_type_pack<U>();
    // check all work group dimensions
    for_all_combinations<invoke_init_reduce_over_group>(
        Dims, RetType, ReducedType, Operators, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
}

#endif
