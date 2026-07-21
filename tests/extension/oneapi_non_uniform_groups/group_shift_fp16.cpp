/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_shift.h"

namespace non_uniform_groups::tests {

TEMPLATE_LIST_TEST_CASE("Non-uniform-group shift",
                        "[oneapi_non_uniform_groups][group_func][fp16]",
                        GroupPackTypes) {
  auto queue = once_per_unit::get_queue();

  if (queue.get_device().has(sycl::aspect::fp16)) {
    for_all_combinations<shift_non_uniform_group_test>(
        TestType{}, unnamed_type_pack<sycl::half>{}, queue);
  } else {
    WARN("Device does not support half precision floating point operations.");
  }
}

}  // namespace non_uniform_groups::tests
