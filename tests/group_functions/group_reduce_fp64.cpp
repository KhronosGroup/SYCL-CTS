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
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
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

using DoubleExtendedTypes = concatenation<ReduceTypes, double>::type;
// 2-dim Cartesian product of type lists
using prod2 =
    product<std::tuple, DoubleExtendedTypes, DoubleExtendedTypes>::type;

// hipSYCL has no implementation over sub-groups
TEMPLATE_TEST_CASE_SIG("Group and sub-group joint reduce functions",
                       "[group_func][fp64][dim]", ((int D), D), 1, 2, 3) {
  auto queue = once_per_unit::get_queue();
  // check dimensions to only print warning once
  if constexpr (D == 1) {
    // FIXME: hipSYCL omission
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
    WARN(
        "hipSYCL has no implementation of "
        "std::iterator_traits<Ptr>::value_type joint_reduce(sub_group g, "
        "Ptr first, Ptr last, BinaryOperation binary_op) over sub-groups. "
        "Skipping the test case.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
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
  if (queue.get_device().has(sycl::aspect::fp64)) {
    joint_reduce_group<D, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
#endif
}

TEMPLATE_LIST_TEST_CASE("Group and sub-group joint reduce functions with init",
                        "[group_func][type_list][fp64][dim]", prod2) {
  auto queue = once_per_unit::get_queue();
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
  // FIXME: ComputeCpp cannot handle cases of different types
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  if constexpr (std::is_same_v<T, U>)
#endif
  {
    if (queue.get_device().has(sycl::aspect::fp64)) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<U, double>) {
        // check all work group dimensions
        init_joint_reduce_group<1, T, U>(queue);
        init_joint_reduce_group<2, T, U>(queue);
        init_joint_reduce_group<3, T, U>(queue);
      }
    } else {
      WARN(
          "Device does not support double precision floating point "
          "operations.");
    }
  }
#endif
}

TEMPLATE_TEST_CASE_SIG("Group and sub-group reduce functions",
                       "[group_func][fp64][dim]", ((int D), D), 1, 2, 3) {
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
  if (queue.get_device().has(sycl::aspect::fp64)) {
    reduce_over_group<D, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
#endif
}

TEMPLATE_LIST_TEST_CASE("Group and sub-group reduce functions with init",
                        "[group_func][type_list][fp64][dim]", prod2) {
  auto queue = once_per_unit::get_queue();
  using T = std::tuple_element_t<0, TestType>;
  using U = std::tuple_element_t<1, TestType>;

  // check types to only print warning once
  if constexpr (std::is_same_v<T, char> && std::is_same_v<U, char>) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp does not implement reduce for unsigned long long int and "
        "long long int. Skipping the test cases.");
    WARN(
        "ComputeCpp cannot handle cases of different types. "
        "Skipping such test cases.");
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
  if (queue.get_device().has(sycl::aspect::fp64)) {
    // FIXME: ComputeCpp cannot handle cases of different types
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    if constexpr (std::is_same_v<T, U>)
#endif
    {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<U, double>) {
        // check all work group dimensions
        init_reduce_over_group<1, T, U>(queue);
        init_reduce_over_group<2, T, U>(queue);
        init_reduce_over_group<3, T, U>(queue);
      }
    }
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
#endif
}
