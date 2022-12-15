/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../../util/extensions.h"
#include "../common/common.h"
#include "queue_shortcuts_usm.h"

namespace queue_shortcuts_usm_fp16 {

using namespace sycl_cts;
using namespace queue_shortcuts_usm;

TEST_CASE("queue shortcuts unified shared memory fp16", "[queue]") {
  auto queue = util::get_cts_object::queue();
  using avaliability =
      util::extensions::availability<util::extensions::tag::fp16>;
  if (!avaliability::check(queue)) {
    WARN(
        "Device does not support half precision floating point operations"
        "Skipping the test case.");
    return;
  }

  check_queue_shortcuts_usm_for_type<sycl::half>{}(queue, "sycl::half");
}

}  // namespace queue_shortcuts_usm_fp16
