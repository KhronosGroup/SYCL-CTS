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

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"

#define TEST_NAME context_constructors

namespace context_constructors__ {
using namespace sycl_cts;

/** tests the constructors for sycl::context
 */
class TEST_NAME : public util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void check_context_after_ctor(sycl::context& context,
                                sycl::device& expectedDevice,
                                util::logger& log) {
    if (context.get_devices().size() != 1) {
      FAIL(log, "context was not constructed correctly (get_devices size)");
    }

    if (context.get_devices()[0] != expectedDevice) {
      FAIL(log, "context was not constructed correctly (device equality)");
    }
  }

  void check_context_after_ctor(sycl::context& context,
                                std::vector<sycl::device>& expectedDevices,
                                util::logger& log) {
    if (context.get_devices().size() != expectedDevices.size()) {
      FAIL(log, "context was not constructed correctly (get_devices size)");
    }

    for (auto& device : context.get_devices()) {
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
  void run(util::logger& log) override {
    cts_async_handler asyncHandler;
    sycl::property_list property_list{};
    {
      /** check default constructor, destructor
          and (const property_list&) constructor
       */
      {
        sycl::context context;
        sycl::context context_prop(property_list);
      }

      /** check (async_handler) and
          (async_handler, const property_list&) constructors
       */
      {
        sycl::context context(asyncHandler);
        sycl::context context_prop(asyncHandler, property_list);
      }

      /** check (const device&) and
          (const device&, const property_list&) constructors
       */
      {
        auto device = util::get_cts_object::device(cts_selector);
        sycl::context context(device);
        sycl::context context_prop(device, property_list);

        check_context_after_ctor(context, device, log);
        check_context_after_ctor(context_prop, device, log);
      }

      /** check (const device&, async_handler) and
          (const device&, async_handler, const property_list&) constructors
       */
      {
        cts_async_handler asyncHandler;
        auto device = util::get_cts_object::device(cts_selector);
        sycl::context context(device, asyncHandler);
        sycl::context context_prop(device, asyncHandler, property_list);

        check_context_after_ctor(context, device, log);
        check_context_after_ctor(context_prop, device, log);
      }

      /** check (const std::vector<device>&) and
          (const std::vector<device>&, const property_list&) constructors
       */
      {
        auto platform = util::get_cts_object::platform(cts_selector);
        auto deviceList = platform.get_devices();
        sycl::context context(deviceList);
        sycl::context context_prop(deviceList, property_list);

        check_context_after_ctor(context, deviceList, log);
        check_context_after_ctor(context_prop, deviceList, log);
      }

      /** check (const std::vector<device>&, async_handler) and
          (const std::vector<device>&, async_handler, const property_list&)
          constructors
       */
      {
        cts_async_handler asyncHandler;
        auto platform = util::get_cts_object::platform(cts_selector);
        auto deviceList = platform.get_devices();
        sycl::context context(deviceList, asyncHandler);
        sycl::context context_prop(deviceList, asyncHandler, property_list);

        check_context_after_ctor(context, deviceList, log);
        check_context_after_ctor(context_prop, deviceList, log);
      }

      /** check (const platform&) and
          (const platform&, const property_list&) constructors
       */
      {
        auto platform = util::get_cts_object::platform(cts_selector);
        sycl::context context(platform);
        sycl::context context_prop(platform, property_list);
      }

      /** check (const platform&, async_handler) and
          (const platform&, async_handler, const property_list&) constructors
       */
      {
        cts_async_handler asyncHandler;
        auto platform = util::get_cts_object::platform(cts_selector);
        sycl::context context(platform, asyncHandler);
        sycl::context context_prop(platform, asyncHandler, property_list);
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

      /** Check throw when empty devices vector
       */
      {
        std::vector<sycl::device> deviceList;
        CHECK_THROWS_MATCHES(
            sycl::context(deviceList), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace context_constructors__ */
