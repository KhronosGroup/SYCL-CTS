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

//  Provide checks that ranged accessor still creates a requisite for the entire
//  underlying buffer

#include "../common/disabled_for_test_case.h"
#include "../common/get_cts_object.h"
#include "catch2/catch_test_macros.hpp"

#include "accessor_common.h"

namespace accessor_requisite_entire_buffer {

// FIXME: re-enable when possibility of a SYCL kernel with an unnamed type is
// implemented in computecpp
DISABLED_FOR_TEST_CASE(ComputeCpp)
("requisite for the entire underlying buffer for sycl::accessor ",
 "[accessor]")({
  auto q = sycl_cts::util::get_cts_object::queue();

  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    SKIP(
        "test is skipped because device doesn't "
        "support usm_shared_allocations");

  constexpr size_t buffer_size = 10;
  constexpr size_t offset_size = 7;
  int data[buffer_size];
  std::iota(data, (data + buffer_size), 0);
  {
    sycl::buffer<int, 1> data_buf(data, sycl::range(buffer_size));
    // create two commands that uses ranaged accessors that access to
    // non-overlapping regions of the same buffer since ranged accessor still
    // creates a requisite for the entire underlying buffer second sommand
    // should execute only after first finishes to check it both commands assign
    // some data to usm check_data and it is expected that second command will
    // do it only after first command even though first command will do it after
    // delay via loop
    int* check_data = sycl::malloc_shared<int>(1, q);

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::target::device>
          acc(data_buf, cgh, sycl::range<1>(offset_size));

      cgh.single_task<class kernel_assign>([=] {
        // to delay assigning expected_val to check_data use a loop that will
        // take some time
        for (int i = 0; i < 1000000; i++) {
          int s = sycl::sqrt(float(i));
          acc[s % offset_size] = accessor_tests_common::expected_val;
        }
        *check_data = accessor_tests_common::expected_val;
      });
    });

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::device> acc(
          data_buf, cgh, sycl::range<1>(1), sycl::id<1>(offset_size));
      cgh.single_task<class kernel_check>([=] {
        *check_data = accessor_tests_common::changed_val;
        auto v = acc[offset_size];
      });
    });
    q.wait_and_throw();
    CHECK(*check_data == accessor_tests_common::changed_val);
  }
});

}  // namespace accessor_requisite_entire_buffer
