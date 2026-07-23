/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_of.h"

// use wide types to exclude truncation of init values
using WideTypes = std::tuple<int32_t, uint32_t, int64_t, uint64_t, float>;

TEMPLATE_LIST_TEST_CASE("Group and sub_group joint of bool functions",
                        "[group_func][type_list][dim]", WideTypes) {
  auto queue = once_per_unit::get_queue();
  // check all work group dimensions
  joint_of_group<1, TestType>(queue);
  joint_of_group<2, TestType>(queue);
  joint_of_group<3, TestType>(queue);
}

TEMPLATE_LIST_TEST_CASE(
    "Group and sub_group of bool functions with predicate functions",
    "[group_func][type_list][dim]", WideTypes) {
  auto queue = once_per_unit::get_queue();
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
  auto queue = once_per_unit::get_queue();
  bool_function_of_group<D>(queue);
  bool_function_of_sub_group<D>(queue);
}
