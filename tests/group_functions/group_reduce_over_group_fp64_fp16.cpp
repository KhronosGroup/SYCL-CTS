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

// need also double tests enabled
#ifdef SYCL_CTS_ENABLE_DOUBLE_TESTS

#include "group_reduce.h"

using ReduceTypes = unnamed_type_pack<double, sycl::half>;

// 2-dim Cartesian product of type lists
using prod2 = product<std::tuple, ReduceTypes, ReduceTypes>::type;

TEMPLATE_LIST_TEST_CASE("Group and sub-group reduce functions with init",
                        "[group_func][fp16][fp64][dim]", prod2) {
  auto queue = once_per_unit::get_queue();
  using T = std::tuple_element_t<0, TestType>;
  using U = std::tuple_element_t<1, TestType>;

  if (queue.get_device().has(sycl::aspect::fp16) &&
      queue.get_device().has(sycl::aspect::fp64)) {
    // Get binary operators from T
    const auto Operators = get_op_types<T>();
    const auto RetType = unnamed_type_pack<T>();
    const auto ReducedType = unnamed_type_pack<U>();
    // check all work group dimensions
    for_all_combinations<invoke_init_reduce_over_group>(
        Dims, RetType, ReducedType, Operators, queue);
  } else {
    WARN(
        "Device does not support half and double precision floating point "
        "operations simultaneously.");
  }
}

#endif
