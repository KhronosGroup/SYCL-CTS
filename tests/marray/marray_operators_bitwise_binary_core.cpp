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

#include "../common/type_coverage.h"
#include "marray_common.h"
#include "marray_operators.h"

namespace marray_operators_bitwise_binary_core {

using namespace marray_operators;

TEST_CASE("bitwise binary operators core", "[marray]") {
  const auto types = marray_common::get_types();
  for_all_types<check_marray_bitwise_binary_operators_for_type>(types);
}

}  // namespace marray_operators_bitwise_binary_core
