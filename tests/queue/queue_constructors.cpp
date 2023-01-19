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

#include "../common/common.h"

namespace queue_constructors {

using namespace sycl_cts;

TEST_CASE("Check queue default constructor and destructor", "[queue]") {
  sycl::queue queue;
}
TEST_CASE("Check queue (property_list) constructor", "[queue]") {
  auto device = util::get_cts_object::device();
  if (!device.has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");

  sycl::queue queue(
      sycl::property_list{sycl::property::queue::enable_profiling()});

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
}

TEST_CASE("Check queue (async_handler) constructor", "[queue]") {
  cts_async_handler asyncHandler;
  sycl::queue queue(asyncHandler);
}

TEST_CASE("Check queue (async_handler, property_list) constructor", "[queue]") {
  auto device = util::get_cts_object::device();
  if (!device.has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  cts_async_handler asyncHandler;
  sycl::queue queue(asyncHandler, {sycl::property::queue::enable_profiling()});

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
}

TEST_CASE("Check queue (device_selector) constructor", "[queue]") {
  cts_selector selector;
  sycl::queue queue(selector);

  CHECK(queue.get_device() == sycl::device(selector));
}

TEST_CASE("Check queue (device_selector, property_list) constructor",
          "[queue]") {
  cts_selector selector;
  if (!sycl::device(selector).has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  sycl::queue queue(selector, {sycl::property::queue::enable_profiling()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
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
  if (!sycl::device(selector).has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  cts_async_handler asyncHandler;
  sycl::queue queue(selector, asyncHandler,
                    {sycl::property::queue::enable_profiling()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
}

TEST_CASE("Check queue (device) constructor", "[queue]") {
  sycl::device device = util::get_cts_object::device();
  sycl::queue queue(device);

  CHECK(queue.get_device() == device);
}

TEST_CASE("Check queue (device, property_list) constructor", "[queue]") {
  sycl::device device = util::get_cts_object::device();
  if (!device.has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  sycl::queue queue(device, {sycl::property::queue::enable_profiling()});

  CHECK(queue.get_device() == device);

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
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
  if (!device.has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  cts_async_handler asyncHandler;
  sycl::queue queue(device, asyncHandler,
                    {sycl::property::queue::enable_profiling()});

  CHECK(queue.get_device() == device);

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
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
  if (!sycl::device(selector).has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  auto context = util::get_cts_object::context(selector);
  sycl::queue queue(context, selector,
                    {sycl::property::queue::enable_profiling()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
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
  if (!sycl::device(selector).has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  auto context = util::get_cts_object::context(selector);
  cts_async_handler asyncHandler;
  sycl::queue queue(context, selector, asyncHandler,
                    {sycl::property::queue::enable_profiling()});

  CHECK(queue.get_device() == sycl::device(selector));

  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());
}

TEST_CASE("Check queue copy constructor", "[queue]") {
  cts_selector selector;
  auto queueA = util::get_cts_object::queue(selector);
  sycl::queue queueB(queueA);

  CHECK(queueA.get_device() == sycl::device(selector));

  CHECK(queueA == queueB);
}

TEST_CASE("Check queue assignment operator", "[queue]") {
  cts_selector selector;
  auto queueA = util::get_cts_object::queue(selector);
  sycl::queue queueB;
  queueB = queueA;

  CHECK(queueA.get_device() == sycl::device(selector));

  CHECK(queueA == queueB);
}

TEST_CASE("Check queue move constructor", "[queue]") {
  cts_selector selector;
  auto queueA = util::get_cts_object::queue(selector);
  auto queueACopy = queueA;
  sycl::queue queueB(std::move(queueA));

  CHECK(queueB == queueACopy);
}

TEST_CASE("Check queue move assignment operator", "[queue]") {
  cts_selector selector;
  auto queueA = util::get_cts_object::queue(selector);
  auto queueACopy = queueA;

  sycl::queue queueB;
  queueB = std::move(queueA);

  CHECK(queueB == queueACopy);
}

TEST_CASE("Check queue equality operator", "[queue]") {
  cts_selector selector;
  auto queueA = util::get_cts_object::queue(selector);
  sycl::queue queueB(queueA);
  sycl::queue queueC(selector);
  queueC = queueA;
  sycl::queue queueD(selector);

  CHECK(queueA == queueB);
  CHECK(queueA == queueC);

  CHECK_FALSE(queueA != queueB);
  CHECK_FALSE(queueA != queueC);

  CHECK_FALSE(queueC == queueD);
  CHECK(queueC != queueD);
}

TEST_CASE("Check queue hashing", "[queue]") {
  cts_selector selector;
  auto queueA = util::get_cts_object::queue(selector);
  sycl::queue queueB(queueA);
  std::hash<sycl::queue> hasher;

  CHECK(hasher(queueA) == hasher(queueB));
}

} /* namespace queue_constructors */
