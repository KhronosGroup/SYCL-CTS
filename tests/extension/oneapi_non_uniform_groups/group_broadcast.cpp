/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_broadcast.h"

namespace non_uniform_groups::tests {

TEMPLATE_LIST_TEST_CASE("Non-uniform group broadcast and select",
                        "[oneapi_non_uniform_groups][group_func][type_list]",
                        GroupPackTypes) {
  auto queue = once_per_unit::get_queue();

  for_all_combinations<broadcast_non_uniform_group_test>(
      TestType{}, CustomTypePack{}, queue);
}

}  // namespace non_uniform_groups::tests
