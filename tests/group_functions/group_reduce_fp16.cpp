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

#include "group_reduce.h"

// FIXME: ComputeCpp does not implement reduce for unsigned long long int and
//        long long int
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
using ReduceTypes =
    unnamed_type_pack<size_t, float, char, signed char, unsigned char,
                      short int, unsigned short int, int, unsigned int,
                      long int, unsigned long int>;
#else
using ReduceTypes = unnamed_type_pack<float, char, int>;
#endif
#else
using ReduceTypes = Types;
#endif

using HalfExtendedTypes = concatenation<ReduceTypes, sycl::half>::type;
// 2-dim Cartesian product of type lists
using prod2 = product<std::tuple, HalfExtendedTypes, HalfExtendedTypes>::type;

// hipSYCL has no implementation over sub-groups
TEST_CASE("Group and sub-group joint reduce functions",
          "[group_func][fp16][dim]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  // FIXME: hipSYCL omission
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  WARN(
      "hipSYCL has no implementation of "
      "std::iterator_traits<Ptr>::value_type joint_reduce(sub_group g, "
      "Ptr first, Ptr last, BinaryOperation binary_op) over sub-groups. "
      "Skipping the test case.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN("ComputeCpp cannot handle half type. Skipping the test.");
#endif

  // FIXME: ComputeCpp has no half
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp16)) {
    // Get binary operators from TestType
    const auto Operators = get_op_types<sycl::half>();
    const auto Type = unnamed_type_pack<sycl::half>();
    for_all_combinations<invoke_joint_reduce_group>(Dims, Type, Operators,
                                                    queue);
  } else {
    WARN("Device does not support half precision floating point operations.");
  }
#endif
}

TEMPLATE_LIST_TEST_CASE("Group and sub-group joint reduce functions with init",
                        "[group_func][type_list][fp16][dim]", prod2) {
  auto queue = sycl_cts::util::get_cts_object::queue();
  using T = std::tuple_element_t<0, TestType>;
  using U = std::tuple_element_t<1, TestType>;

  // check types to only print warning once
  if constexpr (std::is_same_v<T, char> && std::is_same_v<U, char>) {
    // FIXME: hipSYCL omission
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
    WARN(
        "hipSYCL has no implementation of T joint_reduce(sub_group g, Ptr "
        "first, Ptr last, T init, "
        "BinaryOperation binary_op) over sub-groups. Skipping the test case.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp does not implement reduce for unsigned long long int and "
        "long long int. Skipping the test cases.");
    WARN(
        "ComputeCpp cannot handle cases of different types. "
        "Skipping such test cases.");
    WARN("ComputeCpp cannot handle half type. Skipping the test.");
#elif defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    // Link to issue https://github.com/intel/llvm/issues/8341
    WARN(
        "DPCPP cannot handle cases of different types. "
        "Skipping such test cases.");
#endif
  }

  // FIXME: ComputeCpp has no half
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  // FIXME: DPCPP and ComputeCpp cannot handle cases of different types
  // Link to issue https://github.com/intel/llvm/issues/8341
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP) || \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  if constexpr (std::is_same_v<T, U>)
#endif
  {
    if (queue.get_device().has(sycl::aspect::fp16)) {
      if constexpr (std::is_same_v<T, sycl::half> ||
                    std::is_same_v<U, sycl::half>) {
        // Get binary operators from T
        const auto Operators = get_op_types<T>();
        const auto RetType = unnamed_type_pack<T>();
        const auto ReducedType = unnamed_type_pack<U>();
        // check all work group dimensions
        for_all_combinations<invoke_init_joint_reduce_group>(
            Dims, RetType, ReducedType, Operators, queue);
      }
    } else {
      WARN("Device does not support half precision floating point operations.");
    }
  }
#endif
}

TEST_CASE("Group and sub-group reduce functions", "[group_func][fp16][dim]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  // FIXME: ComputeCpp has no half
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
  WARN("ComputeCpp cannot handle half type. Skipping the test.");
#else
  if (queue.get_device().has(sycl::aspect::fp16)) {
    // Get binary operators from TestType
    const auto Operators = get_op_types<sycl::half>();
    const auto Type = unnamed_type_pack<sycl::half>();
    for_all_combinations<invoke_reduce_over_group>(Dims, Type, Operators,
                                                   queue);
  } else {
    WARN("Device does not support half precision floating point operations.");
  }
#endif
}

TEMPLATE_LIST_TEST_CASE("Group and sub-group reduce functions with init",
                        "[group_func][type_list][fp16][dim]", prod2) {
  auto queue = sycl_cts::util::get_cts_object::queue();
  using T = std::tuple_element_t<0, TestType>;
  using U = std::tuple_element_t<1, TestType>;

  // check types to only print warning once
  if constexpr (std::is_same_v<T, char> && std::is_same_v<U, char>) {
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    // Link to issue https://github.com/intel/llvm/issues/8341
    WARN(
        "DPCPP cannot handle cases of different types. "
        "Skipping such test cases.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp does not implement reduce for unsigned long long int and "
        "long long int. Skipping the test cases.");
    WARN(
        "ComputeCpp cannot handle cases of different types. "
        "Skipping such test cases.");
    WARN("ComputeCpp cannot handle half type. Skipping the test.");
#endif
  }

  // FIXME: ComputeCpp has no half
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  if (queue.get_device().has(sycl::aspect::fp16)) {
    // FIXME: DPCPP and ComputeCpp cannot handle cases of different types
    // Link to issue https://github.com/intel/llvm/issues/8341
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP) || \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    if constexpr (std::is_same_v<T, U>)
#endif
    {
      if constexpr (std::is_same_v<T, sycl::half> ||
                    std::is_same_v<U, sycl::half>) {
        // Get binary operators from T
        const auto Operators = get_op_types<T>();
        const auto RetType = unnamed_type_pack<T>();
        const auto ReducedType = unnamed_type_pack<U>();
        // check all work group dimensions
        for_all_combinations<invoke_init_reduce_over_group>(
            Dims, RetType, ReducedType, Operators, queue);
      }
    }
  } else {
    WARN("Device does not support half precision floating point operations.");
  }
#endif
}
