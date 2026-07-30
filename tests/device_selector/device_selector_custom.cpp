/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_selector_custom

namespace device_selector_custom__ {
using namespace sycl_cts;

bool called;

/** tests custom device selectors
 */
int call_selector(const sycl::device &device) {
  called = true;
  return 1;
}

/** create a selector for testing negative scores
 */
class negative_selector {
 public:
  negative_selector(const sycl::device &excludeDevice)
      : excludedDevice{excludeDevice} {}

  int operator()(const sycl::device &device) const {
    return device == excludedDevice ? -1 : 1;
  }

 private:
  const sycl::device excludedDevice;
};

/** check that we can use a custom device selector
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    {
      /** check a custom selector for a device
      */
      {
        called = false;
        auto device = util::get_cts_object::device(call_selector);

        /* check our device selector was used */
        INFO("calling custom selector for creating a device");
        CHECK(called);
      }

      /** check a custom selector for a platform
      */
      {
        called = false;
        auto platform = util::get_cts_object::platform(call_selector);

        /* check our device selector was used */
        INFO("calling custom selector for creating a platform");
        CHECK(called);
      }

      /** check a custom selector for a queue
      */
      {
        called = false;
        auto queue = util::get_cts_object::queue(call_selector);

        /* check our device selector was used */
        INFO("calling custom selector for creating a queue");
        CHECK(called);
      }

      /** check the ability to set negative scores
      */
      {
        auto devices = sycl::device::get_devices();
        if (devices.size() > 1) {
          auto excludedDevice = devices[0];
          negative_selector selector{excludedDevice};
          auto device = util::get_cts_object::device(selector);

          /* check our device selector was used */
          INFO("custom selector selected a device with a negative score");
          CHECK(device != excludedDevice);
        }
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_selector_custom__ */
