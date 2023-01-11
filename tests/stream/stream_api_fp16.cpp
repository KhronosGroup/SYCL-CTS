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

#include "stream_api_common.h"

#define TEST_NAME stream_api_fp16

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class test_kernel;

/** test sycl::stream interface
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
      // Check stream operator for sycl::half
      {
        auto testQueue = util::get_cts_object::queue();

        if (!testQueue.get_device().has(sycl::aspect::fp16)) {
          log.note(
              "Device does not support half precision floating point "
              "operations");
          return;
        }

        testQueue.submit([&](sycl::handler &cgh) {
          sycl::stream os(2048, 80, cgh);

          cgh.single_task<class test_kernel>(
              [=]() { check_all_vec_dims(os, sycl::half(0.2f)); });
        });

        testQueue.wait_and_throw();
      }
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
