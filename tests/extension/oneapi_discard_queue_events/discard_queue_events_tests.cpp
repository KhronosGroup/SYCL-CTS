/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
#include "../../../util/sycl_exceptions.h"
#include "../../../util/usm_helper.h"
#include "../../common/common.h"
#include <cassert>

namespace discard_queue_events::tests {
void negative_test_for_event(sycl::property_list properties) {
  auto device = sycl_cts::util::get_cts_object::device();
  auto queue = sycl::queue(device, properties);

  // kernel submission should return invalid event
  auto e = queue.submit([](sycl::handler& cgh) { cgh.single_task([] {}); });

  auto get_wait_list_check = [&e] { e.get_wait_list(); };
  {
    INFO(
        "Check that call get_wait_list() on invalid event throws an exception");
    CHECK_THROWS_MATCHES(get_wait_list_check(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto wait_check = [&e] { e.wait(); };
  {
    INFO("Check that call wait() on invalid event throws an exception");
    CHECK_THROWS_MATCHES(wait_check(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto static_wait_check = [&e] { sycl::event::wait({e}); };
  {
    INFO("Check that call static wait() on invalid event throws an exception");
    CHECK_THROWS_MATCHES(static_wait_check(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto wait_and_throw_check = [&e] { e.wait_and_throw(); };
  {
    INFO(
        "Check that call wait_and_throw() on invalid event throws an "
        "exception");
    CHECK_THROWS_MATCHES(wait_and_throw_check(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto static_wait_and_throw_check = [&e] { sycl::event::wait_and_throw({e}); };
  {
    INFO(
        "Check that call static wait_and_throw() on invalid event throws an "
        "exception");
    CHECK_THROWS_MATCHES(static_wait_and_throw_check(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto sts = e.get_info<sycl::info::event::command_execution_status>();

  {
    INFO(
        "Check return type of "
        "e.get_info<sycl::info::event::command_execution_status>() is "
        "sycl::info::event_command_status");
    CHECK(std::is_same_v<decltype(sts), sycl::info::event_command_status>);
  }
  {
    INFO(
        "Check return value of "
        "e.get_info<sycl::info::event::command_execution_status>() is "
        "sycl::info::event_command_status::ext_oneapi_unknown");
    CHECK(sts == sycl::info::event_command_status::ext_oneapi_unknown);
  }
}

void negative_test_for_handler(sycl::property_list properties) {
  auto device = sycl_cts::util::get_cts_object::device();
  auto queue = sycl::queue(device, properties);

  // kernel submission should return invalid event
  auto e = queue.submit([](sycl::handler& cgh) { cgh.single_task([] {}); });

  auto depends_on_event_check = [&] {
    queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(e);
      cgh.single_task([] {});
    });
  };
  {
    INFO(
        "Check that call handler.depends_on() with invalid event throws an "
        "exception");
    CHECK_THROWS_MATCHES(depends_on_event_check(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto depends_on_events_check = [&] {
    queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on({e});
      cgh.single_task([] {});
    });
  };
  {
    INFO(
        "Check that call handler.depends_on() with vector including invalid "
        "event(s) throws an exception");
    CHECK_THROWS_MATCHES(depends_on_events_check(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
}

void kernel_execution_test(sycl::property_list properties) {
  auto device = sycl_cts::util::get_cts_object::device();
  auto queue = sycl::queue(device, properties);

  if (device.has(sycl::aspect::usm_shared_allocations)) {
    // keep kernel completion status in USM memory, use a unique pointer to
    // prevent a memory leak if an exception occurs during test execution
    auto kernel_exec_sts_unique =
        usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, bool>(queue,
                                                                        1);

    // unique pointer can't be captured by lambda kernel because its copy
    // constructor is deleted, use raw pointer in lambda kernel body
    bool* kernel_exec_sts = kernel_exec_sts_unique.get();

    *kernel_exec_sts = false;

    queue.submit([&](sycl::handler& cgh) {
      cgh.single_task([=] { *kernel_exec_sts = true; });
    });
    queue.wait();
    CHECK(*kernel_exec_sts);

    *kernel_exec_sts = false;

    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(1, [=](sycl::item<1> item) { *kernel_exec_sts = true; });
    });
    queue.wait();
    CHECK(*kernel_exec_sts);

    *kernel_exec_sts = false;

    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range(sycl::range(1), sycl::range(1)),
          [=](sycl::nd_item<1> nd_item) { *kernel_exec_sts = true; });
    });
    queue.wait();
    CHECK(*kernel_exec_sts);

#ifdef SYCL_EXT_ONEAPI_ASSERT
    *kernel_exec_sts = false;

    queue.submit([&](sycl::handler& cgh) {
      cgh.single_task([=] {
        assert(true);
        *kernel_exec_sts = true;
      });
    });
    queue.wait();
    CHECK(*kernel_exec_sts);
#endif  // SYCL_EXT_ONEAPI_ASSERT

    *kernel_exec_sts = false;

    queue.submit([&](sycl::handler& cgh) {
      sycl::stream os(10, 10, cgh);
      cgh.single_task([=] { *kernel_exec_sts = true; });
    });
    queue.wait();
    CHECK(*kernel_exec_sts);
  }

  bool kernel_exec_sts{false};
  {
    sycl::buffer<bool, 1> sts_buf(&kernel_exec_sts, 1);
    queue.submit([&](sycl::handler& cgh) {
      auto sts_acc = sts_buf.template get_access<sycl::access_mode::write>(cgh);
      cgh.single_task([=] { sts_acc[0] = true; });
    });
    queue.wait();
  }
  CHECK(kernel_exec_sts);
}

TEST_CASE("Negative test for invalid event object member functions",
          "[oneapi_discard_queue_events]") {
#ifndef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
  SKIP("SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS is not defined");
#else
  negative_test_for_event(sycl::property_list{
      sycl::ext::oneapi::property::queue::discard_events()});
  negative_test_for_event(
      sycl::property_list{sycl::ext::oneapi::property::queue::discard_events(),
                          sycl::property::queue::in_order()});
#endif  // SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
}

TEST_CASE(
    "Negative test for handler depends_on() member function taking invalid "
    "event",
    "[oneapi_discard_queue_events]") {
#ifndef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
  SKIP("SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS is not defined");
#else
  negative_test_for_handler(sycl::property_list{
      sycl::ext::oneapi::property::queue::discard_events()});
  negative_test_for_handler(
      sycl::property_list{sycl::ext::oneapi::property::queue::discard_events(),
                          sycl::property::queue::in_order()});
#endif  // SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
}

TEST_CASE(
    "Test of simultaneously using discard_events and enable_profiling "
    "properties in queue constructor",
    "[oneapi_discard_queue_events]") {
#ifndef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
  SKIP("SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS is not defined");
#else
  auto device = sycl_cts::util::get_cts_object::device();

  auto check_invalid_use_discard_events = [&] {
    auto queue = sycl::queue(
        device, {sycl::ext::oneapi::property::queue::discard_events(),
                 sycl::property::queue::enable_profiling()});
  };
  CHECK_THROWS_MATCHES(check_invalid_use_discard_events(), sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
#endif  // SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
}

TEST_CASE(
    "Execution test of a kernel submitted to a queue with discard_events "
    "property",
    "[oneapi_discard_queue_events]") {
#ifndef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
  SKIP("SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS is not defined");
#else
  kernel_execution_test(sycl::property_list{
      sycl::ext::oneapi::property::queue::discard_events()});
  kernel_execution_test(
      sycl::property_list{sycl::ext::oneapi::property::queue::discard_events(),
                          sycl::property::queue::in_order()});
#endif  // SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
}
}  // namespace discard_queue_events::tests
