/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2025 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../common/common.h"
#include <future>

namespace queue_flush::tests {

TEST_CASE(
    "the queue flush extension defines the "
    "SYCL_KHR_QUEUE_FLUSH macro",
    "[khr_queue_flush]") {
#ifndef SYCL_KHR_QUEUE_FLUSH
  static_assert(false, "SYCL_KHR_QUEUE_FLUSH is not defined");
#endif
}

TEST_CASE("Flush and spin lock event", "[khr_queue_flush]") {
  sycl::queue q;
  auto e = q.single_task([] {});
  q.khr_flush();
  while (e.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
  };
}

TEST_CASE("Flush and host task", "[khr_queue_flush]") {
  sycl::queue q;
  std::promise<void> promise;

  q.submit([&](sycl::handler& cgh) {
    cgh.host_task([&] { promise.get_future().wait(); });
  });
  q.khr_flush();
  promise.set_value();
  q.wait();
}

}  // namespace queue_flush::tests
