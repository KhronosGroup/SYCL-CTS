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

template <int DIMENSION>
void check_submit() {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto dev = queue.get_device();

  auto max_work_groups_nd =
      dev.get_info<sycl::info::device::max_num_work_groups<DIMENSION>>();
  auto local = sycl::range<DIMENSION>();
  for (int i = 0; i < DIMENSION; i++) {
    local[i] = 1;
    //to avoid running too long
    max_work_groups_nd[i] = std::min(
        max_work_groups_nd[i],
        static_cast<size_t>(std::numeric_limits<unsigned int>::max() + 1));
  }

  int* a = sycl::malloc_shared<int>(1, queue);
  a[0] = 0;
  queue.parallel_for(sycl::nd_range(max_work_groups_nd, local), [=](auto i) {
     if (i.get_global_id(0) == 0) a[0] = 1;
   }).wait_and_throw();

  CHECK(a[0] == 1);

  sycl::free(a, queue);
}

TEST_CASE("max_num_work_groups nd_range submission [khr_max_num_work_groups]") {
  check_submit<1>();
  check_submit<2>();
  check_submit<3>();
}

}  // namespace max_num_work_groups::tests
