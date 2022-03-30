/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::async_handler
//
*******************************************************************************/

#include <catch2/catch_template_test_macros.hpp>

#include "../common/common.h"

using namespace sycl_cts;

struct test_async_handler {
  bool invoked = false;
  void operator()(sycl::exception_list l) {
    invoked = true;
    for (auto &e : l) {
      std::rethrow_exception(e);
    }
  }
};

TEST_CASE("Check that sycl::async_handler is expected type", "[exception]") {
  CHECK(std::is_same_v<sycl::async_handler,
                       std::function<void(sycl::exception_list)>>);
}

TEST_CASE(
    "Check that when there are no exceptions expected then the async handler "
    "is not invoked",
    "[exception]") {
  test_async_handler asyncHandler;
  sycl::queue q(asyncHandler);
  sycl::context ctx(asyncHandler);
  SECTION("queue::wait_and_throw()") {
    q.submit([&](sycl::handler &h) {});
    q.wait_and_throw();
    CHECK_FALSE(asyncHandler.invoked);
  }

  SECTION("queue::throw_asynchronous()") {
    q.submit([&](sycl::handler &h) {});
    q.throw_asynchronous();
    CHECK_FALSE(asyncHandler.invoked);
  }

  SECTION("event::wait_and_throw()") {
    auto event = q.submit([&](sycl::handler &h) {});
    event.wait_and_throw();
    CHECK_FALSE(asyncHandler.invoked);
  }

  SECTION("static event::wait_and_throw(const std::vector<event> &eventList)") {
    auto event = q.submit([&](sycl::handler &h) {});
    std::vector<sycl::event> event_vector(1, event);
    sycl::event::wait_and_throw(event_vector);
    CHECK_FALSE(asyncHandler.invoked);
  }

  SECTION("on queue destruction") {
    { sycl::queue queue(asyncHandler); }
    CHECK_FALSE(asyncHandler.invoked);
  }

  SECTION("on context destruction") {
    { sycl::context context(asyncHandler); }
    CHECK_FALSE(asyncHandler.invoked);
  }

  SECTION("on buffer destruction") {
    {
      sycl::range<1> rng(1);
      sycl::buffer<int> buf(rng, {sycl::property::buffer::context_bound(ctx)});
    }
    CHECK_FALSE(asyncHandler.invoked);
  }
}
