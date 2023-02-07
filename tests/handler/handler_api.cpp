/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

template <unsigned int test_case, unsigned int kernel>
struct kernel_name;

TEST_CASE("handler require()", "[handler]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  const auto range = sycl::range<1>(1);
  int data[1]{0};

  {
    auto buffer = sycl::buffer<int, 1>(range);

    sycl::buffer<int, 1> resultBuf(data, sycl::range<1>(1));
    auto placeholder =
        sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device,
                       sycl::access::placeholder::true_t>(resultBuf);

    queue.submit([&](sycl::handler &cgh) {
      cgh.require(placeholder);

      cgh.single_task<kernel_name<0, 0>>([=] { placeholder[0] = 1; });
    });
  }

  CHECK(data[0] == 1);

  queue.wait_and_throw();
}

TEST_CASE("handler depends_on(event)", "[handler]") {
  auto queue = sycl_cts::util::get_cts_object::queue();

  {
    sycl::buffer<int, 1> buf_e1{sycl::range<1>{1}};
    sycl::event e1 = queue.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{buf_e1, cgh, sycl::write_only};
      cgh.single_task<kernel_name<1, 0>>([=] { acc[0] = 1; });
    });

    sycl::buffer<int, 1> buf_e2{sycl::range<1>{1}};
    sycl::event e2 = queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(e1);
      sycl::accessor acc{buf_e2, cgh, sycl::write_only};
      cgh.single_task<kernel_name<1, 1>>([=] { acc[0] = 1; });
    });

    e2.wait();
    CHECK(sycl::info::event_command_status::complete ==
          e1.get_info<sycl::info::event::command_execution_status>());
  }
}

TEST_CASE("handler depends_on(vector<event>)", "[handler]") {
  auto queue = sycl_cts::util::get_cts_object::queue();

  {
    sycl::buffer<int, 1> buf_e1{sycl::range<1>{1}};
    sycl::event e1 = queue.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{buf_e1, cgh, sycl::write_only};
      cgh.single_task<kernel_name<2, 0>>([=] { acc[0] = 1; });
    });

    sycl::buffer<int, 1> buf_e2{sycl::range<1>{1}};
    sycl::event e2 = queue.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{buf_e2, cgh, sycl::write_only};
      cgh.single_task<kernel_name<2, 1>>([=] { acc[0] = 1; });
    });

    sycl::buffer<int, 1> buf_e3{sycl::range<1>{1}};
    sycl::event e3 = queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on({e1, e2});
      sycl::accessor acc{buf_e3, cgh, sycl::write_only};
      cgh.single_task<kernel_name<2, 2>>([=] { acc[0] = 1; });
    });

    e3.wait();
    CHECK(sycl::info::event_command_status::complete ==
          e1.get_info<sycl::info::event::command_execution_status>());
    CHECK(sycl::info::event_command_status::complete ==
          e2.get_info<sycl::info::event::command_execution_status>());
  }
}
