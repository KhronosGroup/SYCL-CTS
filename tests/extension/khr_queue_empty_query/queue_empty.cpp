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

namespace queue_empty_query::tests {

TEST_CASE(
    "the queue empty query extension defines the "
    "SYCL_KHR_QUEUE_EMPTY_QUERY macro",
    "[khr_queue_empty_query]") {
#ifndef SYCL_KHR_QUEUE_EMPTY_QUERY
  static_assert(false, "SYCL_KHR_QUEUE_EMPTY_QUERY is not defined");
#endif
}

TEST_CASE("queue are empty by default", "[khr_queue_empty_query]") {
  sycl::queue q{};
  CHECK(q.khr_empty());
}

TEST_CASE("queue are not empty when a command have been enqueed",
          "[khr_queue_empty_query]") {
  sycl::queue q{};
  std::promise<void> promise;

  auto e1 = q.submit([&](sycl::handler& cgh) {
    cgh.host_task([&]() { promise.get_future().wait(); });
  });
  CHECK(!q.khr_empty());
  auto e2 = q.single_task(e1, [=] {});
  CHECK(!q.khr_empty());
  promise.set_value();
  e2.wait();
  CHECK(q.khr_empty());
}

}  // namespace queue_empty_query::tests
