/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"

struct stream_kernel {
  void operator()() const {}
};

TEST_CASE("stream_constructors", "[stream]") {
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
}
