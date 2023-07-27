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

#include "../common/disabled_for_test_case.h"
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "group_scan.h"

// FIXME: ComputeCpp does not implement scan for unsigned long long int and long
// long int
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
using ScanTypes =
    unnamed_type_pack<size_t, float, char, signed char, unsigned char,
                      short int, unsigned short int, int, unsigned int,
                      long int, unsigned long int>;
#else
using ScanTypes = unnamed_type_pack<float, char, int>;
#endif
#else
using ScanTypes = Types;
#endif

using Dims = integer_pack<1, 2, 3>;

using TestCombinations1Type = typename get_combinations<Dims, ScanTypes>::type;

// FIXME: hipSYCL cannot handle cases of different types
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
using TestCombinations2Types = TestCombinations1Type;
using TestCombinations3Types = TestCombinations1Type;
#else
using TestCombinations2Types =
    typename get_combinations<Dims, ScanTypes, ScanTypes>::type;
using TestCombinations3Types =
    typename get_combinations<Dims, ScanTypes, ScanTypes, ScanTypes>::type;
#endif
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL
// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Group and sub-group joint scan functions", "[group_func][type_list][dim]",
 TestCombinations2Types)({
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  WARN(
      "hipSYCL cannot handle cases of different types for InPtr and OutPtr. "
      "Skipping such test cases.");
  WARN(
      "hipSYCL joint_exclusive_scan and joint_inclusive_scan cannot process "
      "over several sub-groups simultaneously. Using one sub-group only.");
  WARN("hipSYCL does not support sycl::known_identity_v yet.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN("ComputeCpp does not implement joint scan. Skipping the test.");
#endif

  auto queue = once_per_unit::get_queue();

  // Get the packs from the test combination type.
  using DimensionsPack = std::tuple_element_t<0, TestType>;
  using Type1Pack = std::tuple_element_t<1, TestType>;
  const auto Dimensions = DimensionsPack::generate_unnamed();
  const auto Types1 = Type1Pack{};

  // FIXME: ComputeCpp does not implement joint scan
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
  // FIXME: hipSYCL cannot handle cases of different types
#elif defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  for_all_combinations<invoke_joint_scan_group_same_type>(Dimensions, Types1,
                                                          queue);
#else
  using Type2Pack = std::tuple_element_t<2, TestType>;
  const auto Types2 = Type2Pack{};

  for_all_combinations<invoke_joint_scan_group>(Dimensions, Types1, Types2,
                                                queue);
#endif
});

// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Group and sub-group joint scan functions with init",
 "[group_func][type_list][dim]", TestCombinations3Types)({
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  WARN(
      "hipSYCL cannot handle cases of different types for T, *InPtr and "
      "*OutPtr. Skipping such test cases.");
  WARN(
      "hipSYCL joint_exclusive_scan and joint_inclusive_scan with init values "
      "cannot process over several sub-groups simultaneously. Using one "
      "sub-group only.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN("ComputeCpp does not implement joint scan. Skipping the test.");
#endif

  auto queue = once_per_unit::get_queue();
  // FIXME: ComputeCpp does not implement joint scan

  // Get the packs from the test combination type.
  using DimensionsPack = std::tuple_element_t<0, TestType>;
  using Type1Pack = std::tuple_element_t<1, TestType>;
  const auto Dimensions = DimensionsPack::generate_unnamed();
  const auto Types1 = Type1Pack{};

#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
  // FIXME: hipSYCL cannot handle cases of different types
#elif defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  for_all_combinations<invoke_init_joint_scan_group_same_type>(Dimensions,
                                                               Types1, queue);
#else
  using Type2Pack = std::tuple_element_t<2, TestType>;
  using Type3Pack = std::tuple_element_t<3, TestType>;
  const auto Types2 = Type2Pack{};
  const auto Types3 = Type3Pack{};

  for_all_combinations<invoke_init_joint_scan_group>(Dimensions, Types1, Types2,
                                                     Types3, queue);
#endif
});

// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Group and sub-group scan functions", "[group_func][type_list][dim]",
 TestCombinations1Type)({
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN(
      "ComputeCpp does not implement scan for unsigned long long int and "
      "long long int. Skipping the test cases.");
  WARN(
      "ComputeCpp fails to compile with segfault in the compiler. "
      "Skipping the test.");
#endif

  auto queue = once_per_unit::get_queue();

  // Get the packs from the test combination type.
  using DimensionsPack = std::tuple_element_t<0, TestType>;
  using Type1Pack = std::tuple_element_t<1, TestType>;
  const auto Dimensions = DimensionsPack::generate_unnamed();
  const auto Types1 = Type1Pack{};

  // FIXME: Codeplay ComputeCpp - CE 2.11.0
  //        Device Compiler - clang version 8.0.0  (based on LLVM 8.0.0svn)
  //        clang-8: error: unable to execute command: Segmentation fault
  //        clang-8: error: spirv-ll-tool command failed due to signal
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  for_all_combinations<invoke_scan_over_group>(Dimensions, Types1, queue);
#endif
});

// FIXME: hipSYCL has wrong arguments order for inclusive_scan_over_group: init
// and op are interchanged. known_identity is not implemented yet.
DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Group and sub-group scan functions with init", "[group_func][type_list][dim]",
 TestCombinations2Types)({
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN(
      "ComputeCpp does not implement scan for unsigned long long int and "
      "long long int. Skipping the test cases.");
  WARN(
      "ComputeCpp cannot handle cases of different types for T and V. Skipping "
      "such test cases.");
  WARN(
      "ComputeCpp fails to compile with segfault in the compiler. "
      "Skipping the test.");
#endif

  auto queue = once_per_unit::get_queue();

  // Get the packs from the test combination type.
  using DimensionsPack = std::tuple_element_t<0, TestType>;
  using Type1Pack = std::tuple_element_t<1, TestType>;
  const auto Dimensions = DimensionsPack::generate_unnamed();
  const auto Types1 = Type1Pack{};

  // FIXME: Codeplay ComputeCpp - CE 2.11.0
  //        Device Compiler - clang version 8.0.0  (based on LLVM 8.0.0svn)
  //        clang-8: error: unable to execute command: Segmentation fault
  //        clang-8: error: spirv-ll-tool command failed due to signal
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
  // FIXME: ComputeCpp cannot handle cases of different types
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  for_all_combinations<invoke_init_scan_over_group_same_type>(Dimensions,
                                                              Types1, queue);
#else
  using Type2Pack = std::tuple_element_t<2, TestType>;
  const auto Types2 = Type2Pack{};

  for_all_combinations<invoke_init_scan_over_group>(Dimensions, Types1, Types2,
                                                    queue);
#endif
});
