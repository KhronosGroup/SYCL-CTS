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

#include "group_permute.h"

// hipSYCL does not permute right 8-bit types inside groups
TEMPLATE_LIST_TEST_CASE("Group and sub-group permute",
                        "[group_func][type_list][dim]", CustomTypes) {
  // check types to only print warning once
  if constexpr (std::is_same_v<TestType, char>) {
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL)
    WARN(
        "hipSYCL has not implemented sycl::marray type yet. Skipping the test "
        "cases.");
#elif defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "ComputeCpp does not implement permute functions. "
        "Skipping the test.");
#endif
  }

  // FIXME: ComputeCpp do not implement permute functions
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  return;
#else
  auto queue = sycl_cts::util::get_cts_object::queue();

  // check all work group dimensions
//  permute_group<1, TestType>(queue);
//  permute_group<2, TestType>(queue);
//  permute_group<3, TestType>(queue);

  permute_sub_group<1, TestType>(queue);
  permute_sub_group<2, TestType>(queue);
  permute_sub_group<3, TestType>(queue);
#endif
}
