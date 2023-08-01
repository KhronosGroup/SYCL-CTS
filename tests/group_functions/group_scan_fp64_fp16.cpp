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

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/type_coverage.h"
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "group_scan.h"

using ScanTypes = Types;

using DoubleHalfTypes = unnamed_type_pack<double, sycl::half>;
using DoubleHalfExtendedTypes = concatenation<ScanTypes, DoubleHalfTypes>::type;

using Dims = integer_pack<1, 2, 3>;

using TestCombinations2Types =
    typename get_combinations<Dims, DoubleHalfTypes, DoubleHalfTypes>::type;

// FIXME: hipSYCL cannot handle cases of different types
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
using TestCombinations3Types = TestCombinations2Types;
#else
using TestCombinations3Types =
    typename get_combinations<Dims, DoubleHalfExtendedTypes,
                              DoubleHalfExtendedTypes,
                              DoubleHalfExtendedTypes>::type;
#endif
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL
// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Group and sub-group joint scan functions", "[group_func][fp16][fp64][dim]",
 TestCombinations2Types)({
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  WARN(
      "hipSYCL cannot handle cases of different types for InPtr and OutPtr. "
      "Skipping the test.");
  WARN(
      "hipSYCL joint_exclusive_scan and joint_inclusive_scan cannot process "
      "over several sub-groups simultaneously. Using one sub-group only.");
  WARN("hipSYCL does not support sycl::known_identity_v yet.");
#endif

  // FIXME: hipSYCL cannot handle cases of different types
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
  return;
#else
  auto queue = once_per_unit::get_queue();

  // Get the packs from the test combination type.
  using DimensionsPack = std::tuple_element_t<0, TestType>;
  using Type1Pack = std::tuple_element_t<1, TestType>;
  using Type2Pack = std::tuple_element_t<2, TestType>;
  const auto Dimensions = DimensionsPack::generate_unnamed();
  const auto Types1 = Type1Pack{};
  const auto Types2 = Type2Pack{};

  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    // here DoubleHalfTypes can be used as only half+double combinations are
    // untested
    for_all_combinations_with_two<invoke_joint_scan_group, double, sycl::half>(
        Dimensions, Types1, Types2, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
});

// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Group and sub-group joint scan functions with init",
 "[group_func][type_list][fp16][fp64][dim]", TestCombinations3Types)({
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  WARN(
      "hipSYCL cannot handle cases of different types for T, *InPtr and "
      "*OutPtr. Skipping the test.");
  WARN(
      "hipSYCL joint_exclusive_scan and joint_inclusive_scan with init values "
      "cannot process over several sub-groups simultaneously. Using one "
      "sub-group only.");
#endif

  // FIXME: hipSYCL cannot handle cases of different types
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
  return;
#else
  auto queue = once_per_unit::get_queue();

  // Get the packs from the test combination type.
  using DimensionsPack = std::tuple_element_t<0, TestType>;
  using Type1Pack = std::tuple_element_t<1, TestType>;
  using Type2Pack = std::tuple_element_t<2, TestType>;
  using Type3Pack = std::tuple_element_t<3, TestType>;
  const auto Dimensions = DimensionsPack::generate_unnamed();
  const auto Types1 = Type1Pack{};
  const auto Types2 = Type2Pack{};
  const auto Types3 = Type3Pack{};

  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    for_all_combinations_with_two<invoke_init_joint_scan_group, double,
                                  sycl::half>(Dimensions, Types1, Types2,
                                              Types3, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
});

// FIXME: hipSYCL has wrong arguments order for inclusive_scan_over_group: init
// and op are interchanged. known_identity is not impemented yet.
DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Group and sub-group scan functions with init",
 "[group_func][fp16][fp64][dim]", TestCombinations2Types)({
  auto queue = once_per_unit::get_queue();

  // Get the packs from the test combination type.
  using DimensionsPack = std::tuple_element_t<0, TestType>;
  using Type1Pack = std::tuple_element_t<1, TestType>;
  using Type2Pack = std::tuple_element_t<2, TestType>;
  const auto Dimensions = DimensionsPack::generate_unnamed();
  const auto Types1 = Type1Pack{};
  const auto Types2 = Type2Pack{};

  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    // here DoubleHalfTypes can be used as only half+double combinations are
    // untested
    for_all_combinations_with_two<invoke_init_scan_over_group, double,
                                  sycl::half>(Dimensions, Types1, Types2,
                                              queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
});

#endif
