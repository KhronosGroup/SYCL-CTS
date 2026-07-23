/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

//  Provides tests for multi_ptr explicit conversions for core types

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "../common/disabled_for_test_case.h"
#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_explicit_conversions.h"

namespace multi_ptr_explicit_conversions_core {

TEST_CASE("multi_ptr explicit conversions. Core types", "[multi_ptr]") {
  using namespace multi_ptr_explicit_conversions;
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_multi_ptr_explicit_convert_for_type>(types);
  for_all_types<check_multi_ptr_explicit_convert_for_type>(composite_types);
}

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("generic_ptr alias. Core types", "[multi_ptr]")({
  using namespace multi_ptr_explicit_conversions;
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_generic_ptr_aliases_for_type>(types);
  for_all_types<check_generic_ptr_aliases_for_type>(composite_types);
});
}  // namespace multi_ptr_explicit_conversions_core

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
