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

#define TEST_NAME synchronous_exceptions

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/**
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
    cts_selector selector;
    cl::sycl::queue q(selector);

    try {
      q.submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class TEST_NAME>([=]() {});
      });
    } catch (const cl::sycl::exception &e) {
      // Check methods
      cl::sycl::string_class sc = e.what();
      if (e.has_context()) {
        cl::sycl::context c = e.get_context();
      }
      cl::sycl::cl_int ci = e.get_cl_code();

      log_exception(log, e);
      FAIL(log, "An exception should not really have been thrown");
    }
    q.wait_and_throw();
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // TEST_NAMESPACE
