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

#define TEST_NAME context_constructors

namespace context_constructors__ {
using namespace sycl_cts;

/** tests the constructors for sycl::context
 */
class TEST_NAME : public util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void check_context_after_ctor(sycl::context &context,
                                sycl::device &expectedDevice,
                                util::logger &log) {
    if (context.get_devices().size() != 1) {
      FAIL(log, "context was not constructed correctly (get_devices size)");
    }

    if (context.get_devices()[0] != expectedDevice) {
      FAIL(log, "context was not constructed correctly (device equality)");
    }
  }

  void check_context_after_ctor(sycl::context &context,
                                std::vector<sycl::device> &expectedDevices,
                                util::logger &log) {
    if (context.get_devices().size() != expectedDevices.size()) {
      FAIL(log, "context was not constructed correctly (get_devices size)");
    }

    for (auto &device : context.get_devices()) {
      if (std::find(expectedDevices.begin(), expectedDevices.end(), device) ==
          expectedDevices.end()) {
        FAIL(log,
             "context was not constructed correctly (device not in passed "
             "device list)");
      }
    }
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    cts_async_handler asyncHandler;

    {
      /** check default constructor and destructor
       */
      { sycl::context context; }

      /** check (async_handler) constructor
       */
      { sycl::context context(asyncHandler); }

      /** check (device) constructor
       */
      {
        auto device = util::get_cts_object::device(cts_selector);
        sycl::context context(device);

        check_context_after_ctor(context, device, log);
      }

      /** check (device, async_handler) constructor
       */
      {
        cts_async_handler asyncHandler;
        auto device = util::get_cts_object::device(cts_selector);
        sycl::context context(device, asyncHandler);

        check_context_after_ctor(context, device, log);
      }

      /** check (std::vector<device>) constructor
       */
      {
        auto platform = util::get_cts_object::platform(cts_selector);
        auto deviceList = platform.get_devices();
        sycl::context context(deviceList);

        check_context_after_ctor(context, deviceList, log);
      }

      /** check (std::vector<device>, async_handler) constructor
       */
      {
        cts_async_handler asyncHandler;
        auto platform = util::get_cts_object::platform(cts_selector);
        auto deviceList = platform.get_devices();
        sycl::context context(deviceList, asyncHandler);

        check_context_after_ctor(context, deviceList, log);
      }

      /** check copy constructor
       */
      {
        auto contextA = util::get_cts_object::context(cts_selector);
        sycl::context contextB(contextA);

#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(contextA) !=
              sycl::get_native<sycl::backend::opencl>(contextB)) {
            FAIL(log, "context was not copied correctly");
          }
        }
#endif
      }

      /** check assignment operator
       */
      {
        auto contextA = util::get_cts_object::context(cts_selector);
        sycl::context contextB = contextA;

#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(contextA) !=
              sycl::get_native<sycl::backend::opencl>(contextB)) {
            FAIL(log, "context was not assigned correctly");
          }
        }
#endif
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace context_constructors__ */
