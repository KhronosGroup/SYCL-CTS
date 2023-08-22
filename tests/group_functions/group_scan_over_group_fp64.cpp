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

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "type_coverage.h"
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "group_scan.h"

using ScanTypes = Types;

using DoubleType = unnamed_type_pack<double>;
using DoubleExtendedTypes = concatenation<ScanTypes, double>::type;

static const auto Dims = integer_pack<1, 2, 3>::generate_unnamed();
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL

// FIXME: known_identity is not impemented yet for hipSYCL.
DISABLED_FOR_TEST_CASE(hipSYCL)
("Group and sub-group scan functions", "[group_func][fp64][dim]")({
  auto queue = once_per_unit::get_queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
    for_all_combinations<invoke_scan_over_group>(Dims, DoubleType{}, queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
});

// FIXME: hipSYCL has wrong arguments order for inclusive_scan_over_group: init
// and op are interchanged. known_identity is not impemented yet.
DISABLED_FOR_TEST_CASE(hipSYCL)
("Group and sub-group scan functions with init",
 "[group_func][type_list][fp64][dim]")({
  auto queue = once_per_unit::get_queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
    for_all_combinations_with<invoke_init_scan_over_group, double>(
        Dims, DoubleExtendedTypes{}, DoubleExtendedTypes{}, queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
});
