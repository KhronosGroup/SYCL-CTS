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
#include "../common/disabled_for_test_case.h"
#include "queue_shortcuts_usm.h"

namespace queue_shortcuts_usm_fp64 {

using namespace sycl_cts;
using namespace queue_shortcuts_usm;

// hipSYCL not tested
DISABLED_FOR_TEST_CASE(hipSYCL)
("queue shortcuts unified shared memory fp64", "[queue]")({
  auto queue = util::get_cts_object::queue();
  using avaliability =
      util::extensions::availability<util::extensions::tag::fp64>;
  if (!avaliability::check(queue)) {
    WARN("Device does not support double precision floating point operations");
    return;
  }

  check_queue_shortcuts_usm_for_type<double>{}(queue, "double");
})

}  // namespace queue_shortcuts_usm_fp64
