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

namespace queue_shortcuts_usm_fp64 {

using namespace sycl_cts;
using namespace queue_shortcuts_usm;

TEST_CASE("queue shortcuts unified shared memory fp64", "[queue]") {
  auto queue = util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations.");
  }

  check_queue_shortcuts_usm_for_type<double>{}(queue, "double");
}

}  // namespace queue_shortcuts_usm_fp64
