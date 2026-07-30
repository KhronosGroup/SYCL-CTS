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
  sycl::queue q;
  CHECK(q.khr_empty());
}

TEST_CASE("queue are not empty when a command have been enqueed",
          "[khr_queue_empty_query]") {
  sycl::queue q;
  std::promise<void> promise;

  auto e1 = q.submit([&](sycl::handler& cgh) {
    cgh.host_task([&] { promise.get_future().wait(); });
  });
  CHECK(!q.khr_empty());
  auto e2 = q.single_task(e1, [=] {});
  CHECK(!q.khr_empty());
  promise.set_value();
  e2.wait();
  CHECK(q.khr_empty());
}

}  // namespace queue_empty_query::tests
