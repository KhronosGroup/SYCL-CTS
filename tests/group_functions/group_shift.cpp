/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_shift.h"

// errors in AdaptiveCpp with bool and 8-bit types - only in group shifts
TEMPLATE_LIST_TEST_CASE("Group and sub-group shift",
                        "[group_func][type_list][dim]", CustomTypes) {
  auto queue = once_per_unit::get_queue();

  shift_sub_group<1, TestType>(queue);
  shift_sub_group<2, TestType>(queue);
  shift_sub_group<3, TestType>(queue);
}
