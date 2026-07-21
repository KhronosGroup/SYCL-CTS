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

// errors in AdaptiveCpp with bool and 8-bit types - only in group shifts
TEMPLATE_LIST_TEST_CASE("Non-uniform-group shift",
                        "[oneapi_non_uniform_groups][group_func][type_list]",
                        GroupPackTypes) {
  auto queue = once_per_unit::get_queue();

  for_all_combinations<shift_non_uniform_group_test>(TestType{},
                                                     CustomTypePack{}, queue);
}

}  // namespace non_uniform_groups::tests
