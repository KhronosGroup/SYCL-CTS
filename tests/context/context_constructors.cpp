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

  /** execute the test
   */
  void run(util::logger &log) override {
    cts_async_handler asyncHandler;

    {
      /** check default constructor and destructor
      */
      {
        sycl::context context;
      }

      /** check (async_handler) constructor
      */
      {
        sycl::context context(asyncHandler);
      }

      /** check (device) constructor
      */
      {
        cts_selector selector;
        auto device = util::get_cts_object::device(selector);
        sycl::context context(device);

        if (context.get_devices().size() != 1) {
          FAIL(log, "context was not constructed correctly (get_devices size)");
        }

        if (context.get_devices()[0] != device) {
          FAIL(log, "context was not constructed correctly (device equality)");
        }
      }

      /** check (device, async_handler) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto device = util::get_cts_object::device(selector);
        sycl::context context(device, asyncHandler);

        if (context.get_devices().size() != 1) {
          FAIL(log, "context was not constructed correctly (get_devices size)");
        }

        if (context.get_devices()[0] != device) {
          FAIL(log, "context was not constructed correctly (device equality)");
        }
      }

      /** check (std::vector<device>) constructor
      */
      {
        cts_selector selector;
        auto platform = util::get_cts_object::platform(selector);
        auto deviceList = platform.get_devices();
        sycl::context context(deviceList);

        if (context.get_devices().size() != deviceList.size()) {
          FAIL(log, "context was not constructed correctly (get_devices size)");
        }

        for (auto &device : context.get_devices()) {
          if (std::find(deviceList.begin(), deviceList.end(), device) ==
              deviceList.end()) {
            FAIL(log,
                 "context was not constructed correctly (device not in passed "
                 "device list)");
          }

          if (std::count(context.get_devices().begin(),
                         context.get_devices().end(), device) != 1) {
            FAIL(log,
                 "context was not constructed correctly (duplicate devices)");
          }
        }
      }

      /** check (std::vector<device>, async_handler) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto platform = util::get_cts_object::platform(selector);
        auto deviceList = platform.get_devices();
        sycl::context context(deviceList, asyncHandler);

        if (context.get_devices().size() != deviceList.size()) {
          FAIL(log, "context was not constructed correctly (get_devices size)");
        }

        for (auto &device : context.get_devices()) {
          if (std::find(deviceList.begin(), deviceList.end(), device) ==
              deviceList.end()) {
            FAIL(log,
                 "context was not constructed correctly (device not in passed "
                 "device list)");
          }

          if (std::count(context.get_devices().begin(),
                         context.get_devices().end(), device) != 1) {
            FAIL(log,
                 "context was not constructed correctly (duplicate devices)");
          }
        }
      }

      /** check (platform) constructor
      */
      {
        cts_selector selector;
        auto platform = util::get_cts_object::platform(selector);
        sycl::context context(platform);

        if (context.get_devices().size() != platform.get_devices().size()) {
          FAIL(log, "context was not constructed correctly (get_devices size)");
        }

        for (auto &device : context.get_devices()) {
          if (std::find(platform.get_devices().begin(),
                        platform.get_devices().end(),
                        device) == platform.get_devices().end()) {
            FAIL(log,
                 "context was not constructed correctly (device not in "
                 "platform)");
          }

          if (std::count(context.get_devices().begin(),
                         context.get_devices().end(), device) != 1) {
            FAIL(log,
                 "context was not constructed correctly (duplicate devices)");
          }
        }
      }

      /** check (platform, async_handler) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto platform = util::get_cts_object::platform(selector);
        sycl::context context(platform, asyncHandler);

        if (context.get_devices().size() != platform.get_devices().size()) {
          FAIL(log, "context was not constructed correctly (get_devices size)");
        }

        for (auto &device : context.get_devices()) {
          if (std::find(platform.get_devices().begin(),
                        platform.get_devices().end(),
                        device) == platform.get_devices().end()) {
            FAIL(log,
                 "context was not constructed correctly (device not in "
                 "platform)");
          }

          if (std::count(context.get_devices().begin(),
                         context.get_devices().end(), device) != 1) {
            FAIL(log,
                 "context was not constructed correctly (duplicate devices)");
          }
        }
      }

      /** check copy constructor
      */
      {
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        sycl::context contextB(contextA);

        if (contextA != contextB) {
          FAIL(log, "context was not copied correctly (equality)");
        }

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
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        sycl::context contextB = contextA;

        if (contextA != contextB) {
          FAIL(log, "context was not assigned correctly (equality)");
        }

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

      /** check move constructor
      */
      {
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        auto contextACopy = contextA;
        sycl::context contextB(std::move(contextA));

        if (contextACopy != contextB) {
          FAIL(log, "context was not move constructed correctly (equality)");
        }
      }

      /** check move assignment operator
      */
      {
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        auto contextACopy = contextA;
        sycl::context contextB = std::move(contextA);

        if (contextACopy != contextB) {
          FAIL(log, "context was not move assigned correctly (equality)");
        }
      }

      /* check equality operator
      */
      {
        cts_selector selector;
        sycl::context contextA = util::get_cts_object::context(selector);
        sycl::context contextB{contextA};
        sycl::context contextC = util::get_cts_object::context(selector);
        contextC = contextA;
        sycl::context contextD = util::get_cts_object::context(selector);

        if (!(contextA == contextB)) {
          FAIL(log,
               "device equality does not work correctly (equality of equal "
               "failed)");
        }
        if (!(contextA == contextC)) {
          check_equality(log, contextA, contextC);
          FAIL(log,
               "device equality does not work correctly (equality of equal "
               "failed)");
        }
        if (contextA != contextB) {
          FAIL(log,
               "context non-equality does not work correctly"
               "(copy constructed)");
        }
        if (contextA != contextC) {
          FAIL(log,
               "context non-equality does not work correctly"
               "(copy assigned)");
        }
        if (contextC == contextD) {
          FAIL(log,
               "context equality does not work correctly"
               "(comparing same)");
        }
        if (!(contextC != contextD)) {
          FAIL(log,
               "context non-equality does not work correctly"
               "(comparing same)");
        }
      }

      /** check hash
      */
      {
        auto contextA = util::get_cts_object::context();
        sycl::context contextB(contextA);
        std::hash<sycl::context> hasher;

        if (hasher(contextA) != hasher(contextB)) {
          FAIL(log,
               "context std::hash does not work correctly. (hashing of equals "
               "failed)");
        }
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace context_constructors__ */
