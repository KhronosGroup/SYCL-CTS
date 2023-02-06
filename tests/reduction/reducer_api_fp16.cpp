/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
#include "reducer_api.h"

#include <string>
// FIXME: re-enable when reducer is fully implemented in hipSYCL, ComputeCpp and
// DPCPP
DISABLED_FOR_TEST_CASE(ComputeCpp, hipSYCL, DPCPP)
("reducer api fp16", "[reducer][fp16]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  using avaliability = sycl_cts::util::extensions::availability<
      sycl_cts::util::extensions::tag::fp16>;
  if (!avaliability::check(queue)) {
    SKIP("Device does not support half precision floating point operations.");
  }

  using type = sycl::half;
  const std::string type_name("sycl::half");
  check_reducer_subscript<type>{}(queue, type_name);
  check_reducer_identity<type>{}(queue, type_name);
});
