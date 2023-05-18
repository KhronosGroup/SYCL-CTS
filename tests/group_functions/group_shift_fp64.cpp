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

#include "group_shift.h"

TEMPLATE_TEST_CASE_SIG("Group and sub-group shift", "[group_func][fp64][dim]",
                       ((int D), D), 1, 2, 3) {
  // check dimensions to only print warning once
  if constexpr (D == 1) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp does not implement shift functions. "
        "Skipping the test.");
#endif
  }

  // FIXME: ComputeCpp do not implement shift functions
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  auto queue = once_per_unit::get_queue();

  if (queue.get_device().has(sycl::aspect::fp64)) {
    shift_sub_group<D, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
#endif
}
