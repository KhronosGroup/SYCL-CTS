/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::async_handler
//
*******************************************************************************/

#include <catch2/catch_template_test_macros.hpp>

#include "../common/common.h"

namespace exceptions_async_handler {

int priorities;

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

struct queue_async_handler {
  inline static int exceptions_handled;
  void operator()(sycl::exception_list l) {
    priorities = 2;
    for (auto &e_ptr : l) {
      exceptions_handled++;
      try {
        std::rethrow_exception(e_ptr);
      } catch (const sycl::exception &e) {
        CHECK((e.code() == sycl::errc::accessor ||
               e.code() == sycl::errc::nd_range));
      }
    }
  }
};

struct context_async_handler {
  inline static int exceptions_handled;
  void operator()(sycl::exception_list l) {
    priorities = 1;
    for (auto &e_ptr : l) {
      exceptions_handled++;
      try {
        std::rethrow_exception(e_ptr);
      } catch (const sycl::exception &e) {
        CHECK((e.code() == sycl::errc::accessor ||
               e.code() == sycl::errc::nd_range));
      }
    }
  }
};

TEST_CASE("Check that sycl::async_handler is expected type", "[exception]") {
  CHECK(std::is_same_v<sycl::async_handler,
                       std::function<void(sycl::exception_list)>>);
}

TEST_CASE(
    "Check that, when there is no exception expected, the async handler "
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
    std::vector event_vector{event};
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
TEST_CASE("Priorities of async handlers", "[exception]") {
  priorities = 0;
  queue_async_handler::exceptions_handled = 0;
  context_async_handler::exceptions_handled = 0;
  queue_async_handler qHandler;
  context_async_handler cHandler;
  auto device = util::get_cts_object::device(cts_selector);
  sycl::context context(device, cHandler);

  SECTION("Сheck that queue's handler is used first") {
    sycl::queue q(context, device, qHandler);

    q.submit([&](sycl::handler &cgh) {
      cgh.host_task([=] { throw sycl::exception(sycl::errc::accessor); });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.host_task([=] { throw sycl::exception(sycl::errc::nd_range); });
    });

    q.wait_and_throw();

    CHECK(priorities == 2);
    CHECK(queue_async_handler::exceptions_handled == 2);
  }

  SECTION("Check that context's handler is used if queue doesn't have one") {
    sycl::queue q(context, device);

    q.submit([&](sycl::handler &cgh) {
      cgh.host_task([=] { throw sycl::exception(sycl::errc::accessor); });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.host_task([=] { throw sycl::exception(sycl::errc::nd_range); });
    });

    q.wait_and_throw();

    CHECK(priorities == 1);
    CHECK(context_async_handler::exceptions_handled == 2);
  }
}
}  //  namespace exceptions_async_handler
