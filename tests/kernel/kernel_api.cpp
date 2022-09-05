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

#define TEST_NAME kernel_api

struct kernel_name_api {
  void operator()() const {}
};

namespace kernel_api__ {
using namespace sycl_cts;

/** test sycl::kernel
 */
class TEST_NAME : public sycl_cts::util::test_base {
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
      auto ctsQueue = util::get_cts_object::queue();
      auto deviceList = ctsQueue.get_context().get_devices();
      auto ctx = ctsQueue.get_context();

      // Create kernel
      using k_name = kernel_name_api;
      auto kb =
          sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
      auto kernel = kb.get_kernel(sycl::get_kernel_id<k_name>());
      ctsQueue.submit(
          [&](sycl::handler &h) { h.single_task(k_name{}); });
      ctsQueue.wait_and_throw();

      // Check get_context()
      auto cxt = kernel.get_context();
      check_return_type<sycl::context>(log, cxt, "sycl::kernel::get_context()");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_api__ */
