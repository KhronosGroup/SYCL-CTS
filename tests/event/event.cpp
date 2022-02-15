/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/common.h"

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

// TODO: Can we unify with common/common_by_reference.h and other approaches
// (e.g. accessor tests) into a cohesive, Catch2-idiomatic solution?
TEST_CASE("event provides commmon reference semantics", "[event]") {
  SECTION("copy constructor") {
    STATIC_CHECK(std::is_copy_constructible_v<sycl::event>);
    const auto a = make_device_event();
    const sycl::event b(a);
    CHECK(b == a);
  }

  SECTION("copy assignment operator") {
    STATIC_CHECK(std::is_copy_assignable_v<sycl::event>);
    const auto a = make_device_event();
    auto b = make_device_event();
    b = a;
    CHECK(b == a);
  }

  SECTION("destructor") { STATIC_CHECK(std::is_destructible_v<sycl::event>); }

  SECTION("move constructor") {
    STATIC_CHECK(std::is_move_constructible_v<sycl::event>);
    auto a = make_device_event();
    sycl::event b(std::move(a));
    SUCCEED("event was move-constructed");
  }

  SECTION("move assignment operator") {
    STATIC_CHECK(std::is_move_assignable_v<sycl::event>);
    auto a = make_device_event();
    sycl::event b;
    b = std::move(a);
    SUCCEED("event was move-assigned");
  }

  SECTION("equality operators") {
    const auto a = make_device_event();
    const sycl::event b(a);
    auto c = make_device_event();
    c = a;
    const auto d = make_device_event();

    CHECK(a == b);
    CHECK(a == c);
    CHECK_FALSE(a != b);
    CHECK_FALSE(a != c);
    CHECK_FALSE(c == d);
    CHECK(c != d);
  }

  SECTION("std::hash") {
    const auto a = make_device_event();
    const auto b = make_device_event();
    const sycl::event c = a;
    std::hash<sycl::event> hasher;
    CHECK(hasher(a) != hasher(b));
    CHECK(hasher(a) == hasher(c));
  }
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

/**
 * Encapsulates a host task that waits until resolved (= a boolean flag is set).
 */
class resolvable_host_event {
 public:
  /**
   * @param dependencies An optional list of events to depend on.
   */
  resolvable_host_event(const std::vector<sycl::event>& dependencies = {}) {
    event = util::get_cts_object::queue().submit(
        [this, &dependencies](sycl::handler& cgh) {
          for (auto& dep : dependencies) {
            cgh.depends_on(dep);
          }
          cgh.host_task([this] {
            std::unique_lock<std::mutex> lk(mut);
            cv.wait(lk, [this] { return should_resolve; });
          });
        });
  }

  sycl::event& get_sycl_event() { return event; }

  void resolve() {
    std::lock_guard<std::mutex> lk(mut);
    should_resolve = true;
    cv.notify_one();
  }

  virtual ~resolvable_host_event() {
    resolve();
    event.wait();
  }

 private:
  std::mutex mut;
  std::condition_variable cv;
  bool should_resolve = false;
  sycl::event event;
};

TEST_CASE("event::get_wait_list returns a list of all direct dependencies",
          "[event]") {
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
    future = std::async(std::launch::async, [this, delay]() {
      std::this_thread::sleep_for(delay);
      resolve();
      resolved = true;
    });
  }

  bool did_resolve() const { return resolved; }

 private:
  std::future<void> future;
  std::atomic_bool resolved = false;
};

TEST_CASE("event can be waited upon", "[event]") {
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
      : queue{cts_selector{},
              [this](sycl::exception_list el) { capture(std::move(el)); }} {}
  sycl::queue& get_queue() { return queue; }

  bool has(const std::string& name) const {
    return captured_exceptions.count(name) != 0;
  }

  size_t count() const { return captured_exceptions.size(); }

  void clear() { captured_exceptions.clear(); }

 private:
  sycl::queue queue;
  std::unordered_set<std::string> captured_exceptions;

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

// TODO SPEC: It is unclear whether uncaught exceptions inside
//            host tasks get reported as asynchronous errors.
// See also: https://github.com/KhronosGroup/SYCL-Docs/issues/214
static sycl::event make_throwing_host_event(
    sycl::queue& queue, std::string name,
    const std::vector<sycl::event>& dependencies = {}) {
  return queue.submit([&name, &dependencies](sycl::handler& cgh) {
    for (auto& dep : dependencies) {
      cgh.depends_on(dep);
    }
    cgh.host_task([&name](auto) { throw test_exception{std::move(name)}; });
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
// the async handler associated with the context" means when two queues override
// their (shared) context's async handler.
TEST_CASE(
    "event::wait_and_throw reports asynchronous errors from related events on "
    "other queues",
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

  CHECK(teh2.count() == 2);
  CHECK(teh2.has("some-error"));
  CHECK(teh2.has("another-error"));

  // This should be a no-op as the error already has been consumed
  e1.wait_and_throw();
  CHECK(teh1.count() == 0);
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
  check_get_info_param<sycl::info::event, sycl::info::event_command_status,
                       sycl::info::event::command_execution_status>(
      make_device_event());

  SECTION("for host_task event") {
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
// => Must throw exception w/ ercc::backend_mismatch if querying a paramter for
// a different backend. We can only test this if an implementation supports more
// than one backend.
TODO_TEST_CASE("event::get_backend_info returns backend-specific information",
               "[event]");

template <sycl::info::event_profiling descriptor>
static void check_get_profiling_info_return_type() {
  using paramTraitsType =
      typename sycl::info::param_traits<sycl::info::event_profiling,
                                        descriptor>::return_type;
  CHECK(std::is_same<paramTraitsType, uint64_t>::value);
  CHECK(std::is_same_v<
        decltype(std::declval<sycl::event>().get_profiling_info<descriptor>()),
        uint64_t>);
}

TEST_CASE("event::get_profiling_info works as expected", "[event]") {
  // Check that queries return the expected type.
  check_get_profiling_info_return_type<
      sycl::info::event_profiling::command_submit>();
  check_get_profiling_info_return_type<
      sycl::info::event_profiling::command_start>();
  check_get_profiling_info_return_type<
      sycl::info::event_profiling::command_end>();

  const auto device = sycl::device{cts_selector{}};
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
