/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "group_broadcast.h"

TEMPLATE_TEST_CASE_SIG("Group broadcast", "[group_func][fp64][dim]",
                       ((int D), D), 1, 2, 3) {
  auto queue = once_per_unit::get_queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
    broadcast_group<D, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
}

TEMPLATE_TEST_CASE_SIG("Sub-group broadcast and select",
                       "[group_func][fp64][dim]", ((int D), D), 1, 2, 3) {
  auto queue = once_per_unit::get_queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
    broadcast_sub_group<D, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
}
