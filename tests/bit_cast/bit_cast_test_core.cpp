/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::bit_cast test for generic types.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "bit_cast_test.h"

namespace bit_cast::tests::core {

TEST_CASE("Test sycl::bit_cast, core types", "[bit_cast]") {
  const auto primary_types =
      bit_cast::tests::helper_functions::get_primary_type_pack();
  for_all_combinations<bit_cast::tests::run_bit_cast_test>(primary_types,
                                                           primary_types);
}

}  // namespace bit_cast::tests::core
