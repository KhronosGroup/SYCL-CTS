/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "multi_ptr_subscript_op_common.h"

namespace multi_ptr_subscript_op_fp64 {

TEST_CASE("multi_ptr operator[](std::ptrdiff_t). fp64 type", "[multi_ptr]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  multi_ptr_subscript_op::check_multi_ptr_subscript_op<double>{}("double");
}

}  // namespace multi_ptr_subscript_op_fp64
