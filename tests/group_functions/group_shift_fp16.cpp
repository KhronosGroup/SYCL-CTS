/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_shift.h"

TEMPLATE_TEST_CASE_SIG("Group and sub-group shift", "[group_func][fp16][dim]",
                       ((int D), D), 1, 2, 3) {
  auto queue = once_per_unit::get_queue();

  if (queue.get_device().has(sycl::aspect::fp16)) {
    shift_sub_group<D, sycl::half>(queue);
  } else {
    WARN("Device does not support half precision floating point operations.");
  }
}
