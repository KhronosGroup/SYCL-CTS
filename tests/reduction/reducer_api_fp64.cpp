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
// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "reducer_api.h"
#endif

#include <string>
// FIXME: re-enable when reducer is fully implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reducer api fp64", "[reducer][fp64]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  using avaliability = sycl_cts::util::extensions::availability<
      sycl_cts::util::extensions::tag::fp64>;
  if (!avaliability::check(queue)) {
    SKIP("Device does not support double precision floating point operations.");
  }

  using type = double;
  const std::string type_name("double");
  check_reducer_subscript<type>{}(queue, type_name);
  check_reducer_identity<type>{}(queue, type_name);
});
