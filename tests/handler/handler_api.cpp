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

#define TEST_NAME handler_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

struct simple_struct {
  int a;
  float b;
};

class test_placeholder;

/** tests the API for cl::sycl::handler
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
      auto queue = util::get_cts_object::queue();
      const auto range = cl::sycl::range<1>(1);
      int data[1]{0};

      {
        auto buffer = cl::sycl::buffer<int, 1>(range);

        log.note("Check require() method");
        cl::sycl::buffer<int, 1> resultBuf(data, cl::sycl::range<1>(1));
        auto placeholder =
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                               cl::sycl::access::target::global_buffer,
                               cl::sycl::access::placeholder::true_t>(
                resultBuf);

        queue.submit([&](cl::sycl::handler &cgh) {
          cgh.require(placeholder);

          cgh.single_task<class test_placeholder>(
              [=]() { placeholder[0] = 1; });

        });
      }

      if (data[0] != 1) {
        FAIL(log, "requires method test did not set accessor data correctly");
      }

      queue.wait_and_throw();
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

} /* namespace TEST_NAMESPACE */
