/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

//  Provides tests for multi_ptr explicit conversions for sycl::half type

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "multi_ptr_explicit_conversions.h"

#include "../common/disabled_for_test_case.h"

namespace multi_ptr_explicit_conversions_fp16 {

TEST_CASE("multi_ptr explicit conversions. fp16 type", "[multi_ptr]") {
  using namespace multi_ptr_explicit_conversions;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_explicit_convert_for_type<sycl::half>{}("sycl::half");
}

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("generic_ptr alias. fp16 type", "[multi_ptr]")({
  using namespace multi_ptr_explicit_conversions;
  check_generic_ptr_aliases_for_type<sycl::half>{}("sycl::half");
});
}  // namespace multi_ptr_explicit_conversions_fp16

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
