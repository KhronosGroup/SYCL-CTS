/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2025 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../common/common.h"

namespace max_num_work_groups_macro::tests {
TEST_CASE(
    "the max_num_work_groups extension defines the "
    "SYCL_KHR_MAX_WORK_GROUP_QUERIES macro",
    "[khr_max_num_work_groups]") {
#ifndef SYCL_KHR_MAX_WORK_GROUP_QUERIES
  static_assert(false, "SYCL_KHR_MAX_WORK_GROUP_QUERIES is not defined");
#endif
}
}  // namespace max_num_work_groups_macro::tests
