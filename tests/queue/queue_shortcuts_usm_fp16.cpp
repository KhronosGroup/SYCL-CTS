/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "queue_shortcuts_usm.h"

namespace queue_shortcuts_usm_fp16 {

using namespace sycl_cts;
using namespace queue_shortcuts_usm;

TEST_CASE("queue shortcuts unified shared memory fp16", "[queue]") {
  auto queue = util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations.");
  }

  check_queue_shortcuts_usm_for_type<sycl::half>{}(queue, "sycl::half");
}

}  // namespace queue_shortcuts_usm_fp16
