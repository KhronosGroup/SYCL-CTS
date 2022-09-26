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

#define TEST_NAME device_selector_custom

namespace device_selector_custom__ {
using namespace sycl_cts;

/** tests custom selectors for sycl::device_selector
 */
class call_selector : public sycl::device_selector {
 public:
  mutable bool called;

  call_selector() : called(false) {}

  virtual int operator()(const sycl::device &device) const {
    called = true;
    return 1;
  }
};

/** create a selector for testing negative scores
 */
class negative_selector : public sycl::device_selector {
 public:
  negative_selector(const sycl::device &excludeDevice)
      : excludedDevice{excludeDevice} {}

  virtual int operator()(const sycl::device &device) const {
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
        call_selector selector;
        auto device = util::get_cts_object::device(selector);

        /* check our device selector was used */
        if (!selector.called) {
          FAIL(log, "custom selector was never used for creating a device");
        }
      }

      /** check a custom selector for a platform
      */
      {
        call_selector selector;
        auto platform = util::get_cts_object::platform(selector);

        /* check our device selector was used */
        if (!selector.called) {
          FAIL(log, "custom selector was never used for creating a platform");
        }
      }

      /** check a custom selector for a queue
      */
      {
        call_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        /* check our device selector was used */
        if (!selector.called) {
          FAIL(log, "custom selector was never used for creating a queue");
        }
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
          if (device == excludedDevice) {
            FAIL(log,
                 "custom selector selected a device with a negative score");
          }
        }
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_selector_custom__ */
