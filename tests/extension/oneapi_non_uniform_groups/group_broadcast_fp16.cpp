/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

#include "group_broadcast.h"

namespace non_uniform_groups::tests {

TEMPLATE_LIST_TEST_CASE("Non-uniform group broadcast and select",
                        "[oneapi_non_uniform_groups][group_func][fp16]",
                        GroupPackTypes) {
  auto queue = once_per_unit::get_queue();

  if (queue.get_device().has(sycl::aspect::fp16)) {
    for_all_combinations<broadcast_non_uniform_group_test>(
        TestType{}, unnamed_type_pack<sycl::half>{}, queue);
  } else {
    WARN("Device does not support half precision floating point operations.");
  }
}

}  // namespace non_uniform_groups::tests
