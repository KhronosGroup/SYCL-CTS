/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"

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
  sycl::queue queue(cts_selector);

  CHECK(queue.get_device() == sycl::device(cts_selector));
}

TEST_CASE("Check queue (device_selector, property_list) constructor",
          "[queue]") {
  sycl::queue queue(cts_selector, {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(cts_selector));

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (device_selector, async_handler) constructor",
          "[queue]") {
  cts_async_handler asyncHandler;
  sycl::queue queue(cts_selector, asyncHandler);

  CHECK(queue.get_device() == sycl::device(cts_selector));
}

TEST_CASE(
    "Check queue (device_selector, async_handler, property_list) constructor",
    "[queue]") {
  cts_async_handler asyncHandler;
  sycl::queue queue(cts_selector, asyncHandler,
                    {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(cts_selector));

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
  auto context = util::get_cts_object::context(cts_selector);
  sycl::queue queue(context, cts_selector);

  CHECK(queue.get_device() == sycl::device(cts_selector));
}

TEST_CASE("Check queue (context, device_selector, property_list) constructor",
          "[queue]") {
  auto context = util::get_cts_object::context(cts_selector);
  sycl::queue queue(context, cts_selector, {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(cts_selector));

  CHECK(queue.has_property<sycl::property::queue::in_order>());
}

TEST_CASE("Check queue (context, device_selector, async_handler) constructor",
          "[queue]") {
  auto context = util::get_cts_object::context(cts_selector);
  cts_async_handler asyncHandler;
  sycl::queue queue(context, cts_selector, asyncHandler);

  CHECK(queue.get_device() == sycl::device(cts_selector));
}

TEST_CASE(
    "Check queue (context, device_selector, async_handler, property_list)",
    "[queue]") {
  auto context = util::get_cts_object::context(cts_selector);
  cts_async_handler asyncHandler;
  sycl::queue queue(context, cts_selector, asyncHandler,
                    {sycl::property::queue::in_order()});

  CHECK(queue.get_device() == sycl::device(cts_selector));

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

TEST_CASE("Check exceptions thrown for mismatched context and device",
          "[queue]") {
  const sycl::device device(cts_selector);
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
    CHECK_THROWS_MATCHES(sycl::queue(otherContext, cts_selector),
                         sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  SECTION("constructor (context, deviceSelector, asyncHandler)") {
    cts_async_handler asyncHandler;
    CHECK_THROWS_MATCHES(sycl::queue(otherContext, cts_selector, asyncHandler),
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
}

TEST_CASE("Check that the default context contains all devices", "[queue]") {
  sycl::platform platform{};
  sycl::context defaultContext = sycl::queue{}.get_context();
  CHECK(defaultContext.get_devices() == platform.get_devices());
}

TEST_CASE("Check that queue constructors use the correct context", "[queue]") {
  const sycl::property_list& propList = {};
  cts_async_handler asyncHandler;
  const auto& deviceSelector = sycl::default_selector_v;
  sycl::device syclDevice;
  sycl::context syclContext;
  sycl::context defaultContext = sycl::queue{}.get_context();

  // Check that a default-constructed context is not the default context.
  // recall, explicitly created contexts should not equal the default one.
  CHECK(syclContext != defaultContext);

  // Default context constructors
  CHECK(defaultContext == sycl::queue{propList}.get_context());
  CHECK(defaultContext == sycl::queue{asyncHandler, propList}.get_context());
  CHECK(defaultContext == sycl::queue{deviceSelector, propList}.get_context());
  CHECK(defaultContext ==
        sycl::queue{deviceSelector, asyncHandler, propList}.get_context());
  CHECK(defaultContext == sycl::queue{syclDevice, propList}.get_context());
  CHECK(defaultContext ==
        sycl::queue{syclDevice, asyncHandler, propList}.get_context());

  // Non-default context constructors
  CHECK(syclContext ==
        sycl::queue{syclContext, deviceSelector, propList}.get_context());
  CHECK(syclContext ==
        sycl::queue{syclContext, deviceSelector, asyncHandler, propList}
            .get_context());
  CHECK(syclContext ==
        sycl::queue{syclContext, syclDevice, propList}.get_context());
  CHECK(syclContext ==
        sycl::queue{syclContext, syclDevice, asyncHandler, propList}
            .get_context());
}

} /* namespace queue_constructors */
