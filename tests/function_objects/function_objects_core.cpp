/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/type_coverage.h"

// fixme: re-enable when sycl::marray is implemented in AdaptiveCpp
#ifndef SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "function_objects.h"
#endif

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("function objects void specializations scalar core", "[function_objects]")({
  const auto types = named_type_pack<TYPES>::generate(TYPE_NAMES);
  for_all_combinations<check_scalar_return_type>(get_op_types(), types, types);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("function objects void specializations vector core", "[function_objects]")({
  const auto types_vector =
      named_type_pack<TYPES_VECTOR>::generate(TYPE_NAMES_VECTOR);
  for_all_combinations<check_vector_return_type>(get_op_types(), types_vector);
});
