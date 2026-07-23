/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "multi_ptr_subscript_op_common.h"

namespace multi_ptr_subscript_op_core {

TEST_CASE("multi_ptr operator[](std::ptrdiff_t)", "[multi_ptr]") {
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();

  for_all_types<multi_ptr_subscript_op::check_multi_ptr_subscript_op>(types);
  for_all_types<multi_ptr_subscript_op::check_multi_ptr_subscript_op>(
      composite_types);
}

}  // namespace multi_ptr_subscript_op_core
