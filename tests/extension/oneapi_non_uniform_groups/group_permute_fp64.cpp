/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_permute.h"

namespace non_uniform_groups::tests {

TEMPLATE_LIST_TEST_CASE("Non-uniform-group permute",
                        "[oneapi_non_uniform_groups][group_func][fp64]",
                        GroupPackTypes) {
  auto queue = once_per_unit::get_queue();

  if (queue.get_device().has(sycl::aspect::fp64)) {
    for_all_combinations<permute_non_uniform_group_test>(
        TestType{}, unnamed_type_pack<double>{}, queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
}

}  // namespace non_uniform_groups::tests
