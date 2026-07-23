/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

//  Provides tests for multi_ptr explicit conversions for double type

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "multi_ptr_explicit_conversions.h"

#include "../common/disabled_for_test_case.h"

namespace multi_ptr_explicit_conversions_fp64 {

TEST_CASE("multi_ptr explicit conversions. fp64 type", "[multi_ptr]") {
  using namespace multi_ptr_explicit_conversions;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_explicit_convert_for_type<double>{}("double");
}

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("generic_ptr alias. fp64 type", "[multi_ptr]")({
  using namespace multi_ptr_explicit_conversions;
  check_generic_ptr_aliases_for_type<double>{}("double");
});

}  // namespace multi_ptr_explicit_conversions_fp64

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
