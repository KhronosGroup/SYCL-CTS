/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

//  Provides tests multi_ptr convert assignment operators for core types

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_convert_assignment_ops.h"

namespace multi_ptr_convert_assignment_ops_core {

TEST_CASE("Convert assignment operators. core types", "[multi_ptr]") {
  using namespace multi_ptr_convert_assignment_ops;
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_multi_ptr_convert_assign_for_type>(types);
  for_all_types<check_multi_ptr_convert_assign_for_type>(composite_types);
}

}  // namespace multi_ptr_convert_assignment_ops_core

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
