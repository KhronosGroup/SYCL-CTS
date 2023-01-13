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
#include "marray_alignment.h"
#include "marray_common.h"

namespace marray_alignment_fp16 {

using namespace sycl_cts;
using namespace marray_alignment;

TEST_CASE("alignment fp16", "[marray]") {
  auto queue = util::get_cts_object::queue();
  using avaliability =
      util::extensions::availability<util::extensions::tag::fp16>;
  if (!avaliability::check(queue)) {
    WARN(
        "Device does not support half precision floating point operations."
        "Skipping the test case.");
    return;
  }

  check_marray_alignment_for_type<sycl::half>{}("sycl::half");
}

}  // namespace marray_alignment_fp16
