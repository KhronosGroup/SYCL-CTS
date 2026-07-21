/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_broadcast.h"

using BroadcastTypes = CustomTypes;

TEMPLATE_LIST_TEST_CASE("Group broadcast", "[group_func][type_list][dim]",
                        BroadcastTypes) {
  auto queue = once_per_unit::get_queue();
  // check all work group dimensions
  broadcast_group<1, TestType>(queue);
  broadcast_group<2, TestType>(queue);
  broadcast_group<3, TestType>(queue);
}

TEMPLATE_LIST_TEST_CASE("Sub-group broadcast and select",
                        "[group_func][type_list][dim]", BroadcastTypes) {
  auto queue = once_per_unit::get_queue();
  // check all work group dimensions
  broadcast_sub_group<1, TestType>(queue);
  broadcast_sub_group<2, TestType>(queue);
  broadcast_sub_group<3, TestType>(queue);
}
