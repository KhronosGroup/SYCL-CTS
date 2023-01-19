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

#include "group_of.h"

// use wide types to exclude truncation of init values
using WideTypes = std::tuple<int32_t, uint32_t, int64_t, uint64_t, float>;

TEMPLATE_LIST_TEST_CASE("Group and sub_group joint of bool functions",
                        "[group_func][type_list][dim]", WideTypes) {
  auto queue = sycl_cts::util::get_cts_object::queue();

  // check all work group dimensions
  joint_of_group<1, TestType>(queue);
  joint_of_group<2, TestType>(queue);
  joint_of_group<3, TestType>(queue);
}

TEMPLATE_LIST_TEST_CASE(
    "Group and sub_group of bool functions with predicate functions",
    "[group_func][type_list][dim]", WideTypes) {
  auto queue = sycl_cts::util::get_cts_object::queue();

  // check all work group dimensions
  predicate_function_of_group<1, TestType>(queue);
  predicate_function_of_group<2, TestType>(queue);
  predicate_function_of_group<3, TestType>(queue);

  predicate_function_of_sub_group<1, TestType>(queue);
  predicate_function_of_sub_group<2, TestType>(queue);
  predicate_function_of_sub_group<3, TestType>(queue);
}

TEMPLATE_TEST_CASE_SIG("Group and sub_group of bool functions",
                       "[group_func][dim]", ((int D), D), 1, 2, 3) {
  auto queue = sycl_cts::util::get_cts_object::queue();

  bool_function_of_group<D>(queue);
  bool_function_of_sub_group<D>(queue);
}
