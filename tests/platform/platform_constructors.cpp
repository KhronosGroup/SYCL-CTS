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

#include "../common/common.h"

#define TEST_NAME platform_constructors

namespace platform_constructors__ {
using namespace sycl_cts;

/** tests the constructors for sycl::platform
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    {
      /** check default constructor and destructor
       */
      {
        sycl::platform platform;
        sycl::device device;
        // Check if the devices platform contain the device returned by the
        // `default_selector_v`
        // Assume that `device` default constructor use `default_selector_v` as
        // by the spec
        auto platform_devices = platform.get_devices();
        if (std::find(platform_devices.begin(), platform_devices.end(),
                      device) == platform_devices.end()) {
          FAIL(log,
               "platform was not constructed correctly (doesn't contain "
               "default)");
        }
      }

      /** check (const device_selector) constructor
       */
      {
        sycl::platform platform(cts_selector);
        sycl::device device(cts_selector);
        const auto platform_devices = platform.get_devices();
        if (std::find(platform_devices.begin(), platform_devices.end(),
                      device) == platform_devices.end()) {
          FAIL(log,
               "platform was not constructed correctly (doesn't contain asked "
               "device)");
        }
      }

      /** check copy constructor
       */
      {
        sycl::platform platformA(cts_selector);
        sycl::platform platformB(platformA);

#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(platformA) !=
              sycl::get_native<sycl::backend::opencl>(platformB)) {
            FAIL(log, "platform was not copy constructed correctly");
          }
        }
#endif
      }

      /** check assignment operator
       */
      {
        sycl::platform platformA(cts_selector);
        sycl::platform platformB = platformA;

#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(platformA) !=
              sycl::get_native<sycl::backend::opencl>(platformB)) {
            FAIL(log, "platform was not copy assigned correctly");
          }
        }
#endif
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace platform_constructors__ */
