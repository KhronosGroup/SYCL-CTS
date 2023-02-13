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
#include "../common/disabled_for_test_case.h"

struct stream_kernel {
  void operator()() const {}
};

TEST_CASE("stream_constructors", "[stream]"){
  auto queue = sycl_cts::util::get_cts_object::queue();
  size_t bufferSize = 2048;
  size_t workItemBufferSize = 80;

  /** check (size_t, size_t, sycl::handler&) constructor
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::stream os(bufferSize, workItemBufferSize, handler);

      if (os.size() != bufferSize) {
        FAIL("sycl::context::size() returned an incorrect value.");
      }
      if (os.get_work_item_buffer_size() != workItemBufferSize) {
        FAIL(
            "sycl::context::get_work_item_buffer_size() returned an "
            "incorrect value.");
      }

      handler.single_task<class kernel_default>(stream_kernel{});
    });
  }

  /** check (size_t, size_t, sycl::handler&, sycl::property_list&)
   * constructor
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::property_list property_list{};
      sycl::stream os(bufferSize, workItemBufferSize, handler, property_list);

      if (os.size() != bufferSize) {
        FAIL("sycl::context::size() returned an incorrect value.");
      }
      if (os.get_work_item_buffer_size() != workItemBufferSize) {
        FAIL(
            "sycl::context::get_work_item_buffer_size() returned an "
            "incorrect value.");
      }

      handler.single_task<class kernel_property>(stream_kernel{});
    });
  }

  queue.wait_and_throw();
};
