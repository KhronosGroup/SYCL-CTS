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

class test_placeholder;

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

      cgh.single_task<class test_placeholder>([=]() { placeholder[0] = 1; });
    });
  }

  CHECK(data[0] == 1);

  queue.wait_and_throw();
}
