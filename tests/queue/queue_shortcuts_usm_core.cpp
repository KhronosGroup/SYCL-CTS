/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "queue_shortcuts_common.h"
#include "queue_shortcuts_usm.h"

namespace queue_shortcuts_usm_core {

using namespace queue_shortcuts_common;
using namespace queue_shortcuts_usm;

TEST_CASE("queue shortcuts unified shared memory core", "[queue]") {
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  const auto types = get_types();
  for_all_types<check_queue_shortcuts_usm_for_type>(types, queue);
}

}  // namespace queue_shortcuts_usm_core
