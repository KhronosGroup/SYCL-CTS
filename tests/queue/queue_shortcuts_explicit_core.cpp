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

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "queue_shortcuts_common.h"
#include "queue_shortcuts_explicit.h"

namespace queue_shortcuts_explicit_core {

using namespace queue_shortcuts_common;
using namespace queue_shortcuts_explict;

// DPCPP does not define the explicit copy operations
DISABLED_FOR_TEST_CASE(DPCPP)
("queue shortcuts explicit copy core", "[queue]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  const auto types = get_types();
  for_all_types<check_queue_shortcuts_explicit_for_type>(types, queue);
})

}  // namespace queue_shortcuts_explicit_core
