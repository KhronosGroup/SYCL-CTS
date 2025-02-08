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
//  Provides sycl::bit_cast test for fp64 type.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "bit_cast_test.h"


namespace bit_cast::tests::fp64 {

TEST_CASE("Test sycl::bit_cast, fp64 type", "[bit_cast]"){
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

  const auto primary_fp64_type = get_cts_types::get_fp64_type();
  const auto primary_core_types =
      bit_cast::tests::helper_functions::get_primary_type_pack();

  for_all_combinations<bit_cast::tests::run_bit_cast_test>(primary_fp64_type,
                                                           primary_core_types);
  for_all_combinations<bit_cast::tests::run_bit_cast_test>(primary_core_types,
                                                           primary_fp64_type);
  for_all_combinations<bit_cast::tests::run_bit_cast_test>(primary_fp64_type,
                                                           primary_fp64_type);
}

}  // namespace bit_cast::tests::fp64
