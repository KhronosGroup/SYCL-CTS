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
