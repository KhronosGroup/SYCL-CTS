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

// DPCPP does not define stream::size nor stream::get_work_item_buffer_size
#ifndef SYCL_CTS_COMPILING_WITH_DPCPP
/**
 * Check equality of two stream objects. Returns true on equal, false
 * otherwise. */
static bool areEqual(sycl::stream &osA, sycl::stream &osB) {
  if (osA.get_work_item_buffer_size() == osB.get_work_item_buffer_size() ||
      osA.get_precision() == osB.get_precision() || osA.size() == osB.size() ||
      osA.get_stream_mode() == osB.get_stream_mode())
    return false;
  return true;
}
#endif  // SYCL_CTS_COMPILING_WITH_DPCPP

DISABLED_FOR_TEST_CASE(DPCPP)
("stream_constructors", "[stream]")({
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

      handler.single_task(stream_kernel{});
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

      handler.single_task(stream_kernel{});
    });
  }

  /** check copy constructor
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::stream osA(bufferSize, workItemBufferSize, handler);
      sycl::stream osB(osA);

      if (osA.get_work_item_buffer_size() != osB.get_work_item_buffer_size()) {
        FAIL(
            "stream is not copy constructed correctly. "
            "(get_work_item_buffer_size)");
      }
      if (osA.get_precision() != osB.get_precision()) {
        FAIL("stream is not copy constructed correctly. (get_precision)");
      }
      if (osA.size() != osB.size()) {
        FAIL("stream is not copy constructed correctly. (size)");
      }
      if (osA.get_stream_mode() != osB.get_stream_mode()) {
        FAIL("stream is not copy constructed correctly. (get_stream_mode)");
      }
      if (osB.size() != bufferSize) {
        FAIL(
            "sycl::context::size() returned an incorrect value "
            "after copy constructed.");
      }
      if (osB.get_work_item_buffer_size() != workItemBufferSize) {
        FAIL(
            "sycl::context::get_work_item_buffer_size() returned an "
            "incorrect value after copy constructed.");
      }

      handler.single_task(stream_kernel{});
    });
  }

  /** check assignment operator
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::stream osA(bufferSize, workItemBufferSize, handler);
      sycl::stream osB(bufferSize / 2, workItemBufferSize / 2, handler);
      osB = osA;

      if (osA.get_work_item_buffer_size() != osB.get_work_item_buffer_size()) {
        FAIL(
            "stream is not copy constructed correctly. "
            "(get_work_item_buffer_size)");
      }
      if (osA.get_precision() != osB.get_precision()) {
        FAIL("stream is not copy constructed correctly. (get_precision)");
      }
      if (osA.size() != osB.size()) {
        FAIL("stream is not copy constructed correctly. (size)");
      }
      if (osA.get_stream_mode() != osB.get_stream_mode()) {
        FAIL("stream is not copy constructed correctly. (get_stream_mode)");
      }
      if (osB.size() != bufferSize) {
        FAIL(
            "sycl::context::size() returned an incorrect value "
            "after copy assigned.");
      }
      if (osB.get_work_item_buffer_size() != workItemBufferSize) {
        FAIL(
            "sycl::context::get_work_item_buffer_size() returned an "
            "incorrect value after copy assigned.");
      }

      handler.single_task(stream_kernel{});
    });
  }

  /* check move constructor
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::stream osA(bufferSize, workItemBufferSize, handler);
      sycl::stream osB(std::move(osA));

      if (osB.size() != bufferSize) {
        FAIL(
            "sycl::context::size() returned an incorrect value "
            "after move constructed.");
      }
      if (osB.get_work_item_buffer_size() != workItemBufferSize) {
        FAIL(
            "sycl::context::get_work_item_buffer_size() returned an "
            "incorrect value after move constructed.");
      }

      handler.single_task(stream_kernel{});
    });
  }

  /* check move assignment operator
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::stream osA(bufferSize, workItemBufferSize, handler);
      sycl::stream osB(bufferSize / 2, workItemBufferSize / 2, handler);
      osB = std::move(osA);

      if (osB.size() != bufferSize) {
        FAIL(
            "sycl::context::size() returned an incorrect value "
            "after move assigned.");
      }
      if (osB.get_work_item_buffer_size() != workItemBufferSize) {
        FAIL(
            "sycl::context::get_work_item_buffer_size() returned an "
            "incorrect value after move assigned.");
      }

      handler.single_task(stream_kernel{});
    });
  }

  /** check equality operator
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::stream osA(bufferSize, workItemBufferSize, handler);
      sycl::stream osB(osA);
      sycl::stream osC(bufferSize * 2, workItemBufferSize * 2, handler);
      osC = osA;
      sycl::stream osD(bufferSize * 2, workItemBufferSize * 2, handler);

      if (!(osA == osB) && areEqual(osA, osB)) {
        FAIL("stream equality does not work correctly (copy constructed)");
      }

      if (!(osA == osC) && areEqual(osA, osC)) {
        FAIL("stream equality does not work correctly (copy assigned)");
      }
      if (osA != osB) {
        FAIL(
            "stream non-equality does not work correctly"
            "(copy constructed)");
      }
      if (osA != osC) {
        FAIL(
            "stream non-equality does not work correctly"
            "(copy assigned)");
      }
      if (osC == osD) {
        FAIL(
            "stream equality does not work correctly"
            "(comparing same)");
      }
      if (!(osC != osD)) {
        FAIL(
            "stream non-equality does not work correctly"
            "(comparing same)");
      }

      handler.single_task(stream_kernel{});
    });
  }

  /** check hashing
   */
  {
    queue.submit([&](sycl::handler &handler) {
      sycl::stream osA(bufferSize, workItemBufferSize, handler);
      sycl::stream osB = osA;

      std::hash<sycl::stream> hasher;

      if (hasher(osA) != hasher(osB)) {
        FAIL(
            "stream hashing does not work correctly (hashing of equal "
            "failed)");
      }
      handler.single_task(stream_kernel{});
    });
  }

  queue.wait_and_throw();
});
