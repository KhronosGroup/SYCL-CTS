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
#include "../common/type_coverage.h"
#include "marray_common.h"
#include "marray_operators.h"

namespace marray_operators_post_unary_fp64 {

using namespace sycl_cts;
using namespace marray_operators;

TEST_CASE("suffix unary operators fp64", "[marray]") {
  auto queue = util::get_cts_object::queue();
  using availability =
      util::extensions::availability<util::extensions::tag::fp64>;
  if (!availability::check(queue)) {
    WARN(
        "Device does not support double precision floating point operations."
        "Skipping the test case.");
    return;
  }

  check_marray_post_unary_operators_for_type<double>{}("double");
}

}  // namespace marray_operators_post_unary_fp64
