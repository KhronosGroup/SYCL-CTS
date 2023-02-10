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

//  Provides tests multi_ptr convert assignment operators for double type

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_convert_assignment_ops.h"

namespace multi_ptr_convert_assignment_ops_fp64 {

TEST_CASE("Convert assignment operators. fp64 type", "[multi_ptr]") {
  using namespace multi_ptr_convert_assignment_ops;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_convert_assign_for_type<double>{}("double");
}

}  // namespace multi_ptr_convert_assignment_ops_fp64

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
