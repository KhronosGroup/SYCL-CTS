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
#include "type_coverage.h"
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "group_scan.h"

// FIXME: ComputeCpp does not implement scan for unsigned long long int and long
// long int
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
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

using DoubleHalfTypes = unnamed_type_pack<double, sycl::half>;
using DoubleHalfExtendedTypes = concatenation<ScanTypes, DoubleHalfTypes>::type;

static const auto Dims = integer_pack<1, 2, 3>::generate_unnamed();
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL
// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEST_CASE(hipSYCL)
("Group and sub-group joint scan functions", "[group_func][fp16][fp64][dim]")({
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  WARN(
      "hipSYCL cannot handle cases of different types for InPtr and OutPtr. "
      "Skipping the test.");
  WARN(
      "hipSYCL joint_exclusive_scan and joint_inclusive_scan cannot process "
      "over several sub-groups simultaneously. Using one sub-group only.");
  WARN("hipSYCL does not support sycl::known_identity_v yet.");
#elif defined(SYCL_CTS_COMPILING_WITH_DPCPP)
  // Link to issue https://github.com/intel/llvm/issues/8341
  WARN(
      "DPCPP cannot handle cases of different types for InPtr and OutPtr. "
      "Skipping the test.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN("ComputeCpp does not implement joint scan. Skipping the test.");
  WARN("ComputeCpp cannot handle half type. Skipping the test.");
#endif

  // FIXME: ComputeCpp does not implement joint scan and half type
  // FIXME: hipSYCL and DPCPP cannot handle cases of different types
  // Link to issue https://github.com/intel/llvm/issues/8341
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL) || \
    defined(SYCL_CTS_COMPILING_WITH_DPCPP) ||   \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    // here DoubleHalfTypes can be used as only half+double combinations are
    // untested
    for_all_combinations_with_two<invoke_joint_scan_group, double, sycl::half>(
        Dims, DoubleHalfTypes{}, DoubleHalfTypes{}, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
});

// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEST_CASE(hipSYCL)
("Group and sub-group joint scan functions with init",
 "[group_func][type_list][fp16][fp64][dim]")({
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
  WARN(
      "hipSYCL cannot handle cases of different types for T, *InPtr and "
      "*OutPtr. Skipping the test.");
  WARN(
      "hipSYCL joint_exclusive_scan and joint_inclusive_scan with init values "
      "cannot process over several sub-groups simultaneously. Using one "
      "sub-group only.");
#elif defined(SYCL_CTS_COMPILING_WITH_DPCPP)
  // Link to issue https://github.com/intel/llvm/issues/8341
  WARN(
      "DPCPP cannot handle cases of different types for T, *InPtr and "
      "*OutPtr. Skipping the test.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN("ComputeCpp does not implement joint scan. Skipping the test.");
  WARN("ComputeCpp cannot handle half type. Skipping the test.");
#endif

  // FIXME: ComputeCpp does not implement joint scan and half type
  // FIXME: hipSYCL and DPCPP cannot handle cases of different types
  // Link to issue https://github.com/intel/llvm/issues/8341
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL) || \
    defined(SYCL_CTS_COMPILING_WITH_DPCPP) ||   \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    for_all_combinations_with_two<invoke_init_joint_scan_group, double,
                                  sycl::half>(Dims, DoubleHalfExtendedTypes{},
                                              DoubleHalfExtendedTypes{},
                                              DoubleHalfExtendedTypes{}, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
});

// FIXME: hipSYCL has wrong arguments order for inclusive_scan_over_group: init
// and op are interchanged. known_identity is not impemented yet.
DISABLED_FOR_TEST_CASE(hipSYCL)
("Group and sub-group scan functions with init",
 "[group_func][fp16][fp64][dim]")({
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP)
  // Link to issue https://github.com/intel/llvm/issues/8341
  WARN(
      "DPCPP cannot handle cases of different types for T and V. Skipping the "
      "test.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN(
      "ComputeCpp cannot handle cases of different types for T and V. Skipping "
      "the test.");
  WARN("ComputeCpp cannot handle half type. Skipping the test.");
#endif

  // FIXME: ComputeCpp has no half
  // FIXME: DPCPP and ComputeCpp cannot handle cases of different types
  // Link to issue https://github.com/intel/llvm/issues/8341
#if defined(SYCL_CTS_COMPILING_WITH_DPCPP) || \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    // here DoubleHalfTypes can be used as only half+double combinations are
    // untested
    for_all_combinations_with_two<invoke_init_scan_over_group, double,
                                  sycl::half>(Dims, DoubleHalfTypes{},
                                              DoubleHalfTypes{}, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
#endif
});

#endif
