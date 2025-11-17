/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/common.h"
#include "event.h"

using namespace sycl_cts;

class event_kernel;

static sycl::event make_device_event(
    const std::vector<sycl::event>& dependencies = {},
    sycl::queue queue = util::get_cts_object::queue()) {
  return queue.submit([&dependencies](sycl::handler& cgh) {
    for (auto& dep : dependencies) {
      cgh.depends_on(dep);
    }
    cgh.single_task<event_kernel>([] {});
  });
}

TEST_CASE("event provides a default constructor", "[event]") {
  STATIC_CHECK(std::is_default_constructible_v<sycl::event>);
  sycl::event event{};

  // A default constructed event is immediately completed.
  const auto status =
      event.get_info<sycl::info::event::command_execution_status>();
  CHECK(status == sycl::info::event_command_status::complete);

  // A default constructed event acts as though it was created from a
  // default-constructed queue.
  sycl::queue default_queue{};
  CHECK(event.get_backend() == default_queue.get_backend());
}

TEST_CASE("event::get_backend returns the associated backend", "[event]") {
  CHECK(std::is_nothrow_invocable_v<decltype(&sycl::event::get_backend),
                                    const sycl::event>);
  const auto e = make_device_event();
  // There isn't really anything more we can test here, as all members of the
  // sycl::backend enum are implementation defined.
  e.get_backend();
  SUCCEED();
}

TEST_CASE("event::get_wait_list returns a list of all direct dependencies",
          "[event]") {
#if SYCL_CTS_COMPILING_WITH_SIMSYCL
  FAIL("SimSYCL does not implement asynchronous execution.");
#endif

  resolvable_host_event e_a;
  resolvable_host_event e_b{{e_a.get_sycl_event()}};
  resolvable_host_event e_c;
  resolvable_host_event e_d{{e_b.get_sycl_event(), e_c.get_sycl_event()}};

  CHECK(std::is_same_v<std::vector<sycl::event>,
                       decltype(e_d.get_sycl_event().get_wait_list())>);
  const auto wait_list = e_d.get_sycl_event().get_wait_list();
  CHECK(wait_list.size() == 2);
  CHECK(std::find(wait_list.cbegin(), wait_list.cend(), e_b.get_sycl_event()) !=
        wait_list.cend());
  CHECK(std::find(wait_list.cbegin(), wait_list.cend(), e_c.get_sycl_event()) !=
        wait_list.cend());

  e_a.resolve();
  e_b.resolve();
  e_c.resolve();
  e_d.resolve();
}

/**
 * A resolvable_host_event that automatically resolves itself after a given
 * delay.
 */
class delayed_host_event : public resolvable_host_event {
 public:
  delayed_host_event(std::chrono::milliseconds delay)
      : resolvable_host_event() {
    future = std::async(std::launch::async, [this, delay] {
      std::this_thread::sleep_for(delay);
      // For the purpose of the tests it's important that `resolved` will be
      // true whenever SYCL event is completed. As such, we have to set this
      // flag before actually resolving the `future` because otherwise the
      // current thread can go to sleep before the flag is set and the checks
      // would be failing.
      resolved = true;
      resolve();
    });
  }

  bool did_resolve() const { return resolved; }

 private:
  std::future<void> future;
  std::atomic_bool resolved = false;
};

TEST_CASE("event can be waited upon", "[event]") {
#if SYCL_CTS_COMPILING_WITH_SIMSYCL
  FAIL("SimSYCL does not implement asynchronous execution.");
#endif

  // Give main thread some time to fail the did_resolve check
  delayed_host_event dhe{std::chrono::milliseconds(100)};

  auto& event = dhe.get_sycl_event();

  SECTION("event::wait()") { event.wait(); }
  SECTION("event::wait_and_throw()") { event.wait_and_throw(); }
  SECTION("event::wait(std::vector<event>)") {
    sycl::event::wait(std::vector{event});
  }
  SECTION("event::wait_and_throw(std::vector<event>)") {
    sycl::event::wait_and_throw(std::vector{event});
  }

  // In case event::wait et al. do not actually wait until resolve() has been
  // called, this check races against the async function and may result in a
  // false negative. In case it does properly wait, it will always be true.
  CHECK(dhe.did_resolve());
}

TEST_CASE("multiple events can be waited upon simultaneously", "[event]") {
#if SYCL_CTS_COMPILING_WITH_SIMSYCL
  FAIL("SimSYCL does not implement asynchronous execution.");
#endif

  // Give main thread some time to fail the did_resolve check
  delayed_host_event dhe1{std::chrono::milliseconds(100)};
  delayed_host_event dhe2{std::chrono::milliseconds(100)};
  auto& event1 = dhe1.get_sycl_event();
  auto& event2 = dhe2.get_sycl_event();

  SECTION("event::wait(std::vector<event>)") {
    sycl::event::wait(std::vector{event1, event2});
  }
  SECTION("event::wait_and_throw(std::vector<event>)") {
    sycl::event::wait_and_throw(std::vector{event1, event2});
  }

  CHECK(dhe1.did_resolve());
  CHECK(dhe2.did_resolve());
}

struct test_exception {
  std::string name;
};

class test_exception_handler {
 public:
  test_exception_handler()
      : queue{cts_selector,
              [this](sycl::exception_list el) { capture(std::move(el)); }} {}
  sycl::queue& get_queue() { return queue; }

  bool has(const std::string& name) const {
    return captured_exceptions.count(name) != 0;
  }

  size_t count() const { return captured_exceptions.size(); }

  void clear() { captured_exceptions.clear(); }

 private:
  std::unordered_set<std::string> captured_exceptions;
  // Queue has to be destroyed first since that can trigger the exception
  // handler.
  sycl::queue queue;

  void capture(sycl::exception_list el) {
    for (auto& e : el) {
      try {
        std::rethrow_exception(e);
      } catch (test_exception& te) {
        captured_exceptions.insert(te.name);
      }
    }
  }
};

// According to section 4.10.1 "Any uncaught exception thrown during the
// execution of a host task will be turned into an async error [...]".
static sycl::event make_throwing_host_event(
    sycl::queue& queue, std::string name,
    const std::vector<sycl::event>& dependencies = {}) {
  return queue.submit([name, &dependencies](sycl::handler& cgh) {
    for (auto& dep : dependencies) {
      cgh.depends_on(dep);
    }
    cgh.host_task([name] { throw test_exception{name}; });
  });
}

TEST_CASE("event::wait does not report asynchronous errors", "[event]") {
  test_exception_handler teh;
  auto e = make_throwing_host_event(teh.get_queue(), "some-error");

  SECTION("event::wait") { e.wait(); }
  SECTION("event::wait(std::vector<event>)") {
    sycl::event::wait(std::vector{e});
  }

  CHECK(teh.count() == 0);

  // Queue destruction does not flush unconsumed exceptions so do it manually.
  e.wait_and_throw();
}

TEST_CASE("event::wait_and_throw reports asynchronous errors", "[event]") {
  test_exception_handler teh;
  auto e = make_throwing_host_event(teh.get_queue(), "some-error");

  SECTION("event::wait_and_throw") { e.wait_and_throw(); }
  SECTION("event::wait_and_throw(std::vector<event>)") {
    sycl::event::wait_and_throw(std::vector{e});
  }

  CHECK(teh.count() == 1);
  CHECK(teh.has("some-error"));
}

TEST_CASE(
    "event::wait_and_throw reports asynchronous errors from related events",
    "[event]") {
  test_exception_handler teh;
  auto e1 = make_throwing_host_event(teh.get_queue(), "some-error");
  auto e2 = make_throwing_host_event(teh.get_queue(), "another-error", {e1});

  SECTION("event::wait_and_throw") { e2.wait_and_throw(); }
  SECTION("event::wait_and_throw(std::vector<event>)") {
    sycl::event::wait_and_throw(std::vector{e2});
  }

  // This should be a no-op as the error already has been consumed
  e1.wait_and_throw();

  CHECK(teh.count() == 2);
  CHECK(teh.has("some-error"));
  CHECK(teh.has("another-error"));
}

// TODO SPEC: It is unclear what "any unconsumed error [...] will be passed to
// the async handler associated with the context" means when the context does
// not have an async handler, but the queues do. This test assumes that the
// asynch handler for the queues will be used as they are higher priority for
// async errors.
// See also: https://github.com/KhronosGroup/SYCL-Docs/issues/299
TEST_CASE(
    "event::wait_and_throw reports asynchronous errors from related events on "
    "corresponding queues",
    "[event][todo-spec][!mayfail]") {
  test_exception_handler teh1;
  test_exception_handler teh2;
  REQUIRE(teh1.get_queue() != teh2.get_queue());

  auto e1 = make_throwing_host_event(teh1.get_queue(), "some-error");
  auto e2 = make_throwing_host_event(teh2.get_queue(), "another-error", {e1});

  SECTION("event::wait_and_throw") { e2.wait_and_throw(); }
  SECTION("event::wait_and_throw(std::vector<event>)") {
    sycl::event::wait_and_throw(std::vector{e2});
  }

  CHECK(teh2.count() == 1);
  CHECK(teh2.has("another-error"));

  CHECK(teh1.count() == 1);
  CHECK(teh1.has("some-error"));
}

TEST_CASE("event::wait_and_throw only reports unconsumed asynchronous errors",
          "[event]") {
  test_exception_handler teh;
  make_throwing_host_event(teh.get_queue(), "some-error").wait_and_throw();
  teh.clear();

  auto e = make_throwing_host_event(teh.get_queue(), "another-error");

  SECTION("event::wait_and_throw") { e.wait_and_throw(); }
  SECTION("event::wait_and_throw(std::vector<event>)") {
    sycl::event::wait_and_throw(std::vector{e});
  }

  CHECK(teh.count() == 1);
  CHECK(teh.has("another-error"));
}

TEST_CASE("event::get_info returns correct command execution status",
          "[event]") {
  // First check that return value is of expected type
  check_get_info_param<sycl::info::event::command_execution_status,
                       sycl::info::event_command_status>(make_device_event());

  SECTION("for host_task event") {
#if SYCL_CTS_COMPILING_WITH_SIMSYCL
    FAIL("SimSYCL does not implement asynchronous execution.");
#endif

    resolvable_host_event rhe;
    auto& event = rhe.get_sycl_event();

    const auto status1 =
        event.get_info<sycl::info::event::command_execution_status>();
    CHECK((status1 == sycl::info::event_command_status::submitted ||
           status1 == sycl::info::event_command_status::running));

    rhe.resolve();
    event.wait();

    const auto status2 =
        event.get_info<sycl::info::event::command_execution_status>();
    CHECK(status2 == sycl::info::event_command_status::complete);
  }

  SECTION("for device kernel event") {
    // Here we cannot really control the execution status,
    // so we cannot be as thorough
    auto e1 = make_device_event();
    auto e2 = make_device_event({e1});
    e2.wait();
    // Since e2's CGF depends on e1, the latter must have completed by now
    const auto status =
        e1.get_info<sycl::info::event::command_execution_status>();
    CHECK(status == sycl::info::event_command_status::complete);
  }
}

// TODO: Figure out if/how we want to test this.
// => Must throw exception w/ errc::backend_mismatch if querying a parameter
// for a different backend. We can only test this if an implementation supports
// more than one backend.
TODO_TEST_CASE("event::get_backend_info returns backend-specific information",
               "[event]");

template <typename descriptor>
static void check_get_profiling_info_return_type() {
  CHECK(std::is_same<typename descriptor::return_type, uint64_t>::value);
  CHECK(std::is_same_v<
        decltype(std::declval<sycl::event>().get_profiling_info<descriptor>()),
        uint64_t>);
}

// FIXME: reenable when struct information descriptors are implemented
TEST_CASE("event::get_profiling_info works as expected", "[event]") {
  // Check that queries return the expected type.
  check_get_profiling_info_return_type<
      sycl::info::event_profiling::command_submit>();
  check_get_profiling_info_return_type<
      sycl::info::event_profiling::command_start>();
  check_get_profiling_info_return_type<
      sycl::info::event_profiling::command_end>();

  const auto device = sycl::device{cts_selector};
  if (!device.has(sycl::aspect::queue_profiling)) {
    WARN(
        "Skipping test because device does not have "
        "sycl::aspect::queue_profiling");
    return;
  }

  sycl::queue queue =
      sycl::queue(device, {sycl::property::queue::enable_profiling()});

  auto event = make_device_event({}, queue);
  event.wait();

  const auto submit_time =
      event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  const auto start_time =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end_time =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();

  // While the returned values are implementation defined, we can still
  // perform some basic sanity checks.
  CHECK(submit_time <= start_time);
  CHECK(start_time <= end_time);
}
