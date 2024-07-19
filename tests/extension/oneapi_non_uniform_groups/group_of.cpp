/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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
