/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

namespace queue_constructors {

using namespace sycl_cts;

TEST_CASE("Check queue default constructor and destructor", "[queue]") {
  sycl::queue queue;
}

TEST_CASE("Check queue (property_list) constructor", "[queue]") {
  sycl::queue queue(sycl::property_list{sycl::property::queue::in_order()});

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (async_handler) constructor", "[queue]") {
  cts_async_handler asyncHandler;
  sycl::queue queue(asyncHandler);
}

TEST_CASE("Check queue (async_handler, property_list) constructor", "[queue]") {
  cts_async_handler asyncHandler;
  sycl::queue queue(asyncHandler, {sycl::property::queue::in_order()});

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (device_selector) constructor", "[queue]") {
  cts_selector selector;
  sycl::queue queue(selector);

  CHECK(queue.get_device() == sycl::device(selector));
}

TEST_CASE("Check queue (device_selector, property_list) constructor",
          "[queue]") {
  cts_selector selector;
  sycl::queue queue(selector, {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (device_selector, async_handler) constructor",
          "[queue]") {
  cts_selector selector;
  cts_async_handler asyncHandler;
  sycl::queue queue(selector, asyncHandler);

  CHECK(queue.get_device() == sycl::device(selector));
}

TEST_CASE(
    "Check queue (device_selector, async_handler, property_list) constructor",
    "[queue]") {
  cts_selector selector;
  cts_async_handler asyncHandler;
  sycl::queue queue(selector, asyncHandler,
                    {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (device) constructor", "[queue]") {
  sycl::device device = util::get_cts_object::device();
  sycl::queue queue(device);

  CHECK(queue.get_device() == device);
}

TEST_CASE("Check queue (device, property_list) constructor", "[queue]") {
  sycl::device device = util::get_cts_object::device();
  sycl::queue queue(device, {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == device);

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (device, async_handler) constructor", "[queue]") {
  sycl::device device = util::get_cts_object::device();
  cts_async_handler asyncHandler;
  sycl::queue queue(device, asyncHandler);

  CHECK(queue.get_device() == device);
}

TEST_CASE("Check queue (device, async_handler, property_list) constructor",
          "[queue]") {
  sycl::device device = util::get_cts_object::device();
  cts_async_handler asyncHandler;
  sycl::queue queue(device, asyncHandler, {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == device);

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (context, device_selector) constructor", "[queue]") {
  cts_selector selector;
  auto context = util::get_cts_object::context(selector);
  sycl::queue queue(context, selector);

  CHECK(queue.get_device() == sycl::device(selector));
}

TEST_CASE("Check queue (context, device_selector, property_list) constructor",
          "[queue]") {
  cts_selector selector;
  auto context = util::get_cts_object::context(selector);
  sycl::queue queue(context, selector, {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (context, device_selector, async_handler) constructor",
          "[queue]") {
  cts_selector selector;
  auto context = util::get_cts_object::context(selector);
  cts_async_handler asyncHandler;
  sycl::queue queue(context, selector, asyncHandler);

  CHECK(queue.get_device() == sycl::device(selector));
}

TEST_CASE(
    "Check queue (context, device_selector, async_handler, property_list)",
    "[queue]") {
  cts_selector selector;
  auto context = util::get_cts_object::context(selector);
  cts_async_handler asyncHandler;
  sycl::queue queue(context, selector, asyncHandler,
                    {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (context, device) constructor", "[queue]") {
  sycl::device device = util::get_cts_object::device();
  sycl::context context = util::get_cts_object::context();
  sycl::queue queue(context, device);

  CHECK(queue.get_device() == device);
  CHECK(queue.get_context() == context);
}

TEST_CASE("Check queue (context, device, property_list) constructor",
          "[queue]") {
  sycl::device device = util::get_cts_object::device();
  sycl::context context = util::get_cts_object::context();
  sycl::queue queue(context, device, {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == device);
  CHECK(queue.get_context() == context);

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (context, device, async_handler) constructor",
          "[queue]") {
  sycl::device device = util::get_cts_object::device();
  sycl::context context = util::get_cts_object::context();
  cts_async_handler asyncHandler;
  sycl::queue queue(context, device, asyncHandler);

  CHECK(queue.get_device() == device);
  CHECK(queue.get_context() == context);
}

TEST_CASE("Check queue (context, device, async_handler, property_list)",
          "[queue]") {
  sycl::device device = util::get_cts_object::device();
  sycl::context context = util::get_cts_object::context();
  cts_async_handler asyncHandler;
  sycl::queue queue(context, device, asyncHandler,
                    {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == device);
  CHECK(queue.get_context() == context);

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

// FIXME: re-enable when sycl::errc is implemented in computecpp
DISABLED_FOR_TEST_CASE(ComputeCpp)
("Check exceptions thrown for mismatched context and device", "[queue]")({
  cts_selector selector;
  const sycl::device device(selector);
  sycl::device otherDevice = device;
  auto platforms = sycl::platform::get_platforms();
  for (auto p : platforms) {
    auto devices = p.get_devices();
    for (auto d : devices) {
      if (d != device) {
        otherDevice = d;
        break;
      }
    }
    if (otherDevice != device) break;
  }
  if (otherDevice == device) SKIP("No other root device is available");

  sycl::context otherContext(otherDevice);

  SECTION("constructor (context, deviceSelector)") {
    CHECK_THROWS_MATCHES(sycl::queue(otherContext, selector), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  SECTION("constructor (context, deviceSelector, asyncHandler)") {
    cts_async_handler asyncHandler;
    CHECK_THROWS_MATCHES(sycl::queue(otherContext, selector, asyncHandler),
                         sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  SECTION("constructor (context, device)") {
    CHECK_THROWS_MATCHES(sycl::queue(otherContext, device), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  SECTION("constructor (context, device, asyncHandler)") {
    cts_async_handler asyncHandler;
    CHECK_THROWS_MATCHES(sycl::queue(otherContext, device, asyncHandler),
                         sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
});

} /* namespace queue_constructors */
