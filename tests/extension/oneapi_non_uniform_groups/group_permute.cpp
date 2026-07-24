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

// AdaptiveCpp does not permute right 8-bit types inside groups
TEMPLATE_LIST_TEST_CASE("Non-uniform-group permute",
                        "[oneapi_non_uniform_groups][group_func][type_list]",
                        GroupPackTypes) {
  auto queue = once_per_unit::get_queue();

  for_all_combinations<permute_non_uniform_group_test>(TestType{},
                                                       CustomTypePack{}, queue);
}

}  // namespace non_uniform_groups::tests
