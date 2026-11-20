/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2025 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/disabled_for_test_case.h"
#include <future>

namespace queue_empty_query::tests {

TEST_CASE(
    "the queue empty query extension defines the "
    "SYCL_KHR_QUEUE_EMPTY_QUERY macro",
    "[khr_queue_empty_query]") {
#ifdef SYCL_KHR_QUEUE_EMPTY_QUERY
  STATIC_REQUIRE(SYCL_KHR_QUEUE_EMPTY_QUERY == 1);
#else
  SKIP("SYCL_KHR_QUEUE_EMPTY_QUERY is not defined");
#endif
}

#ifdef SYCL_KHR_QUEUE_EMPTY_QUERY

TEST_CASE("queue are empty by default", "[khr_queue_empty_query]") {
  sycl::queue q;
  CHECK(q.khr_empty());
}

// AdaptiveCpp does not provide sycl::handler::host_task, which this test
// case uses to keep a command pending while the queue is queried.
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("queue are not empty when a command have been enqueed",
 "[khr_queue_empty_query]")({
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
});

#endif  // SYCL_KHR_QUEUE_EMPTY_QUERY

}  // namespace queue_empty_query::tests
