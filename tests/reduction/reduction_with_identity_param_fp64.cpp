/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for double.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "reduction_with_identity_param.h"
#endif

namespace reduction_with_identity_param_fp64 {

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_with_identity_param_fp64", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations");
  }
  reduction_with_identity_param::run_test_for_type<double>()(queue, "double");
});

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_with_identity_param_item_twice_fp64", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations");
  }
  reduction_with_identity_param::run_test_for_type_item_twice<double>()(
      queue, "double");
});

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_with_identity_param_even_item_fp64", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations");
  }
  reduction_with_identity_param::run_test_for_type_even_item<double>()(
      queue, "double");
});

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_with_identity_param_no_one_fp64", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations");
  }
  reduction_with_identity_param::run_test_for_type_no_one_item<double>()(
      queue, "double");
});
}  // namespace reduction_with_identity_param_fp64
