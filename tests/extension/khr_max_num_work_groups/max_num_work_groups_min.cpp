/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
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

#include "../../common/common.h"

namespace max_num_work_groups::tests {

template <int N>
void check_min_size() {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto dev = queue.get_device();
  if (dev.get_info<sycl::info::device::device_type>() !=
      sycl::info::device_type::custom) {
    auto max_work_groups =
        dev.get_info<khr::info::device::max_num_work_groups<N>>();

    for (int i = 0; i < N; i++) {
      CHECK(max_work_groups[i] >= 1);
    }
  }
}

TEST_CASE("if the device is not info::device_type::custom, the minimal",
          "size in each dimension is (1,1,1)", "[khr_max_num_work_groups]") {
  check_min_size<1>();
  check_min_size<2>();
  check_min_size<3>();
}

}  // namespace max_num_work_groups::tests
