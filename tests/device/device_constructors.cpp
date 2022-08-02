/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
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

#define TEST_NAME device_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::device
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
    try {
      /** check default constructor and destructor
       */
      {
        cl::sycl::device device;

        if (!device.is_host()) {
          FAIL(log, "device was not constructed correctly (is_host)");
        }
      }

      /** check (device_selector) constructor
       */
      {
        cts_selector selector;
        cl::sycl::device device(selector);

        if (device.is_host() != selector.is_host()) {
          FAIL(log, "device was not constructed correctly (is_host)");
        }
      }

      /** check copy constructor
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(deviceA);

        if (deviceA.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not copied correctly (is_host)");
        }

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && deviceA.get() != deviceB.get()) {
          FAIL(log, "device was not assigned correctly (get)");
        }
#endif
      }

      /** check assignment operator
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB = deviceA;

        if (deviceA.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not assigned correctly (is_host)");
        }
#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && deviceA.get() != deviceB.get()) {
          FAIL(log, "device was not assigned correctly (get)");
        }
#endif
      }

      /** check move constructor
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(std::move(deviceA));

        if (selector.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not move constructed correctly (is_host)");
        }
      }

      /** check move assignment operator
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB = std::move(deviceA);

        if (selector.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not move assigned correctly (is_host)");
        }
      }

      /* check equality operator
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(deviceA);
        cl::sycl::device deviceC(selector);
        deviceC = deviceA;

        if (!(deviceA == deviceB)) {
          FAIL(log,
               "device equality does not work correctly (copy constructed)");
        }
        if (!(deviceA == deviceC)) {
          FAIL(log, "device equality does not work correctly (copy assigned)");
        }
        if (deviceA != deviceB) {
          FAIL(log,
               "device non-equality does not work correctly"
               "(copy constructed)");
        }
        if (deviceA != deviceC) {
          FAIL(log,
               "device non-equality does not work correctly"
               "(copy assigned)");
        }
      }

      /* check hash
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(deviceA);
        cl::sycl::device deviceC = deviceA;

        cl::sycl::hash_class<cl::sycl::device> hasher;

        if (hasher(deviceA) != hasher(deviceB)) {
          FAIL(log,
               "device hash_class does not work correctly (copy constructed)");
        }
        if (hasher(deviceA) != hasher(deviceC)) {
          FAIL(log,
               "device hash_class does not work correctly (copy assigned)");
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_constructors__ */
