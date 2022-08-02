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

#define TEST_NAME context_api

namespace context_api__ {
using namespace sycl_cts;

/** tests the api for cl::sycl::context
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
    try {
      auto context = util::get_cts_object::context();

      /** check is_host() method
       */
      {
        auto isHost = context.is_host();
        check_return_type<bool>(log, isHost, "is_host()");
      }

      /** check get_devices() method
       */
      {
        auto deviceList = context.get_devices();
        check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
            log, deviceList, "get_devices()");
      }

      /** check get_platform() method
       */
      {
        auto platform = context.get_platform();
        check_return_type<cl::sycl::platform>(log, platform, "get_platform()");
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

} /* namespace context_api */
