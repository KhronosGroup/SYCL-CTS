/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

#include "group_permute.h"

namespace non_uniform_groups::tests {

TEST_CASE("Non-uniform-group permute",
          "[oneapi_non_uniform_groups][group_func][fp64]") {
  auto queue = once_per_unit::get_queue();

  if (queue.get_device().has(sycl::aspect::fp64)) {
    permute_non_uniform_group<oneapi_ext::ballot_group<sycl::sub_group>,
                              double>(queue);
    permute_non_uniform_group<oneapi_ext::fixed_size_group<1, sycl::sub_group>,
                              double>(queue);
    permute_non_uniform_group<oneapi_ext::fixed_size_group<2, sycl::sub_group>,
                              double>(queue);
    permute_non_uniform_group<oneapi_ext::fixed_size_group<4, sycl::sub_group>,
                              double>(queue);
    permute_non_uniform_group<oneapi_ext::fixed_size_group<8, sycl::sub_group>,
                              double>(queue);
    permute_non_uniform_group<oneapi_ext::tangle_group<sycl::sub_group>,
                              double>(queue);
    permute_non_uniform_group<oneapi_ext::opportunistic_group, double>(queue);
  } else {
    WARN("Device does not support double precision floating point operations.");
  }
}

}  // namespace non_uniform_groups::tests
