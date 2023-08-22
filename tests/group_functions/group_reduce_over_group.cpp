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

#include "group_reduce.h"

using ReduceTypes = Types;

// 2-dim Cartesian product of type lists
using prod2 = product<std::tuple, ReduceTypes, ReduceTypes>::type;

TEMPLATE_LIST_TEST_CASE("Group and sub-group reduce functions",
                        "[group_func][type_list][dim]", ReduceTypes) {
  auto queue = once_per_unit::get_queue();
  // Get binary operators from TestType
  const auto Operators = get_op_types<TestType>();
  const auto Type = unnamed_type_pack<TestType>();
  for_all_combinations<invoke_reduce_over_group>(Dims, Type, Operators, queue);
}

TEMPLATE_LIST_TEST_CASE("Group and sub-group reduce functions with init",
                        "[group_func][type_list][dim]", prod2) {
  auto queue = once_per_unit::get_queue();
  using T = std::tuple_element_t<0, TestType>;
  using U = std::tuple_element_t<1, TestType>;

  // Get binary operators from T
  const auto Operators = get_op_types<T>();
  const auto RetType = unnamed_type_pack<T>();
  const auto ReducedType = unnamed_type_pack<U>();
  // check all work group dimensions
  for_all_combinations<invoke_init_reduce_over_group>(
      Dims, RetType, ReducedType, Operators, queue);
}
