/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_of.h"

namespace non_uniform_groups::tests {

// use wide types to exclude truncation of init values
static const auto wide_types =
    named_type_pack<int32_t, uint32_t, int64_t, uint64_t, float>::generate(
        "int32_t", "uint32_t", "int64_t", "uint64_t", "float");

TEMPLATE_LIST_TEST_CASE("Non-uniform group joint of bool functions",
                        "[oneapi_non_uniform_groups][group_func][type_list]",
                        GroupPackTypes) {
  for_all_combinations<joint_of_group_test>(TestType{}, wide_types);
}

TEMPLATE_LIST_TEST_CASE(
    "Non-uniform group of bool functions with predicate functions",
    "[oneapi_non_uniform_groups][group_func][type_list]", GroupPackTypes) {
  for_all_combinations<predicate_function_of_non_uniform_group_test>(
      TestType{}, wide_types);
}

TEMPLATE_LIST_TEST_CASE("Non-uniform group of bool functions",
                        "[oneapi_non_uniform_groups][group_func]",
                        GroupPackTypes) {
  for_all_combinations<bool_function_of_non_uniform_group_test>(TestType{});
}

}  // namespace non_uniform_groups::tests
