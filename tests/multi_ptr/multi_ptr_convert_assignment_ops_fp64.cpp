/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

//  Provides tests multi_ptr convert assignment operators for double type

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_convert_assignment_ops.h"

namespace multi_ptr_convert_assignment_ops_fp64 {

TEST_CASE("Convert assignment operators. fp64 type", "[multi_ptr]") {
  using namespace multi_ptr_convert_assignment_ops;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_convert_assign_for_type<double>{}("double");
}

}  // namespace multi_ptr_convert_assignment_ops_fp64

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
