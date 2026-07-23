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
("function objects void specializations scalar fp64",
 "[function_objects][fp64]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations.");
  }

  const auto types = named_type_pack<TYPES>::generate(TYPE_NAMES);
  const auto types_only_double = named_type_pack<double>::generate("double");

  // prevent testing duplicate combinations
  for_all_combinations<check_scalar_return_type>(get_op_types(), types,
                                                 types_only_double);
  for_all_combinations<check_scalar_return_type>(get_op_types(),
                                                 types_only_double, types);
  for_all_combinations<check_scalar_return_type>(
      get_op_types(), types_only_double, types_only_double);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("function objects void specializations vector fp64",
 "[function_objects][fp64]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations.");
  }

  const auto types_vector_only_double =
      named_type_pack<sycl::vec<double, 3>, sycl::marray<double, 5>>::generate(
          "vec<double, 3>", "marray<double, 5>");

  for_all_combinations<check_vector_return_type>(get_op_types(),
                                                 types_vector_only_double);
});
