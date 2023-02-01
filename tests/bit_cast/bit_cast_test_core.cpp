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
//  Provides sycl::bit_cast test for generic types.
//
*******************************************************************************/
#include "bit_cast_test.h"

namespace bit_cast::tests::core {

TEST_CASE("Test sycl::bit_cast, core types", "[bit_cast]") {
  const auto primary_types =
      bit_cast::tests::helper_functions::get_primary_type_pack();
  for_all_combinations<bit_cast::tests::run_bit_cast_test>(primary_types,
                                                           primary_types);
}

}  // namespace bit_cast::tests::core
