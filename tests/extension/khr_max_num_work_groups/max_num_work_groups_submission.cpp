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

TEST_CASE("max_num_work_groups submission at numeric_limits int max in ",
          "dimension 1 [khr_max_num_work_groups]") {
  sycl::queue q;
  sycl::device dev = q.get_device();

  auto max_work_groups_nd =
      dev.get_info<khr::info::device::max_num_work_groups<3>>();

  if (max_work_groups_nd[0] >= std::numeric_limits<int>::max()) {
    int* result = sycl::malloc_shared<int>(1, q);
    *result = 0;

    sycl::nd_range<3> ndra({std::numeric_limits<int>::max(), 1, 1}, {1, 1, 1});

        q.submit([&](sycl::handler& h) {
           h.parallel_for(ndra, [=](sycl::nd_item<3> item) {
             if (item.get_global_id(0) == std::numeric_limits<int>::max() - 1)
               *result = 42;
           });
         }).wait();

    CHECK(*result == 42);
    sycl::free(result, q);
  }
}

}  // namespace max_num_work_groups::tests
