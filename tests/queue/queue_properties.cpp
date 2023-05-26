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

#include "../../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

#define TEST_NAME queue_properties

namespace queue_properties {

class kernel1;
class kernel2;
class kernel3;

using namespace sycl_cts;

void check_in_order_prop(const sycl::queue& queue) {
  CHECK(queue.is_in_order());

  REQUIRE(queue.has_property<sycl::property::queue::in_order>());

  auto prop = queue.get_property<sycl::property::queue::in_order>();
  check_return_type<sycl::property::queue::in_order>(
      prop,
      "sycl::queue::has_property<sycl::property::queue::"
      "in_order>()");
}

void check_in_order_functionality(sycl::queue& queue) {
  if (!queue.get_device().has(sycl::aspect::usm_device_allocations))
    SKIP(
        "test for in_order functionality is skipped because device doesn't "
        "support usm_device_allocations");

  bool* data_changed = sycl::malloc_device<bool>(1, queue);
  constexpr size_t buffer_size = 10;
  int loop_array[buffer_size];
  bool result = false;
  {
    sycl::buffer<bool, 1> res_buf(&result, sycl::range<1>(1));
    sycl::buffer<int, 1> loop_buf(loop_array, sycl::range(buffer_size));
    // to garantee that data_changed initialized as false before two tested
    // commands are submitted
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.single_task<kernel1>([=] { *data_changed = false; });
        })
        .wait_and_throw();

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_buf.get_access<sycl::access_mode::write>(cgh);
      auto loop_acc = loop_buf.get_access<sycl::access_mode::read_write>(cgh);

      cgh.single_task<kernel2>([=] {
        // to delay checking data_changed use a loop that will take some time
        for (int i = 0; i < 1000000; i++) {
          int s = sycl::sqrt(float(i));
          loop_acc[s % buffer_size] = i;
        }
        res_acc[0] = (*data_changed == false);
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      cgh.single_task<kernel3>([=] { *data_changed = true; });
    });
    queue.wait_and_throw();
  }
  CHECK(result);
}

void check_in_order(sycl::queue& queue) {
  check_in_order_prop(queue);

  check_in_order_functionality(queue);
}

void check_enable_profiling_prop(sycl::queue& queue) {
  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());

  auto prop = queue.get_property<sycl::property::queue::enable_profiling>();
  check_return_type<sycl::property::queue::enable_profiling>(
      prop,
      "sycl::queue::has_property<sycl::property::queue::"
      "enable_profiling>()");
}

void check_props(sycl::queue& queue) {
  check_enable_profiling_prop(queue);
  check_in_order_prop(queue);
}

void check_in_order_throws(sycl::queue& queue) {
  auto action = [&] {
    auto get_prop =
        queue.template get_property<sycl::property::queue::enable_profiling>();
  };
  CHECK_THROWS_MATCHES(action(), sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
}

void check_enable_profiling_throws(sycl::queue& queue) {
  auto action = [&] {
    auto get_prop =
        queue.template get_property<sycl::property::queue::in_order>();
  };
  CHECK_THROWS_MATCHES(action(), sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
}

TEST_CASE("check property::queue::enable_profiling", "[queue]") {
  auto device = util::get_cts_object::device();
  if (!device.has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");

  sycl::queue queue(
      device, sycl::property_list{sycl::property::queue::enable_profiling()});
  check_enable_profiling_prop(queue);
  check_enable_profiling_throws(queue);
}

TEST_CASE("check property::queue::in_order", "[queue]") {
  cts_async_handler asyncHandler;
  auto context = util::get_cts_object::context(cts_selector);
  auto device = util::get_cts_object::device();

  SECTION("with constructor (propList)") {
    sycl::queue queue(sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION("with constructor (asyncHandler, propList)") {
    sycl::queue queue(asyncHandler,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION("with constructor (deviceSelector, propList)") {
    sycl::queue queue(cts_selector,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION("with constructor (deviceSelector, asyncHandler, propList)") {
    sycl::queue queue(cts_selector, asyncHandler,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION("with constructor (syclDevice, propList)") {
    sycl::queue queue(device,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION("with constructor (syclDevice, asyncHandler, propList)") {
    sycl::queue queue(device, asyncHandler,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION("with constructor (syclContext, deviceSelector, propList)") {
    sycl::queue queue(context, cts_selector,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION(
      "with constructor (syclContext, deviceSelector, asyncHandler, "
      "propList)") {
    sycl::queue queue(context, cts_selector, asyncHandler,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION("with constructor (syclContext, syclDevice, propList)") {
    sycl::queue queue(context, device,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
  SECTION(
      "with constructor (syclContext, syclDevice, asyncHandler, propList)") {
    sycl::queue queue(context, device, asyncHandler,
                      sycl::property_list{sycl::property::queue::in_order()});
    check_in_order(queue);
    check_in_order_throws(queue);
  }
}

TEST_CASE("check both queue properties in_order and enable_profiling",
          "[queue]") {
  auto device = util::get_cts_object::device();
  if (!device.has(sycl::aspect::queue_profiling))
    SKIP("Device does not support queue_profiling");
  cts_async_handler asyncHandler;
  auto context = util::get_cts_object::context(cts_selector);

  SECTION("with constructor (propList)") {
    sycl::queue queue(
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION("with constructor (asyncHandler, propList)") {
    sycl::queue queue(
        asyncHandler,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION("with constructor (deviceSelector, propList)") {
    sycl::queue queue(
        cts_selector,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION("with constructor (deviceSelector, asyncHandler, propList)") {
    sycl::queue queue(
        cts_selector, asyncHandler,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION("with constructor (syclDevice, propList)") {
    sycl::queue queue(
        device, sycl::property_list{sycl::property::queue::in_order(),
                                    sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION("with constructor (syclDevice, asyncHandler, propList)") {
    sycl::queue queue(
        device, asyncHandler,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION("with constructor (syclContext, deviceSelector, propList)") {
    sycl::queue queue(
        context, cts_selector,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION(
      "with constructor (syclContext, deviceSelector, asyncHandler, "
      "propList)") {
    sycl::queue queue(
        context, cts_selector, asyncHandler,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_in_order(queue);
  }
  SECTION("with constructor (syclContext, syclDevice, propList)") {
    sycl::queue queue(
        context, device,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
  SECTION(
      "with constructor (syclContext, syclDevice, asyncHandler, propList)") {
    sycl::queue queue(
        context, device, asyncHandler,
        sycl::property_list{sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling()});
    check_props(queue);
  }
}
}  // namespace queue_properties
