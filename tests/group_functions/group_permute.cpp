/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_permute.h"

// AdaptiveCpp does not permute right 8-bit types inside groups
TEMPLATE_LIST_TEST_CASE("Group and sub-group permute",
                        "[group_func][type_list][dim]", CustomTypes) {
  auto queue = once_per_unit::get_queue();

  permute_sub_group<1, TestType>(queue);
  permute_sub_group<2, TestType>(queue);
  permute_sub_group<3, TestType>(queue);
}
