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
