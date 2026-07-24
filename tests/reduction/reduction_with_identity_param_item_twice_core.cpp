/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for interaction between each work item and reductions twice
//  with identity parameter for arithmetic scalar types.
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "reduction_with_identity_param.h"
#endif

namespace reduction_with_identity_param_item_twice_core {

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_with_identity_param_even_item_twice_core", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  for_all_types<reduction_with_identity_param::run_test_for_type_item_twice>(
      reduction_common::scalar_types, queue);
});
}  // namespace reduction_with_identity_param_item_twice_core
