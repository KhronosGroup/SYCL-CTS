/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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

#define TEST_NAME accessor_constructors_image

#include "../common/common.h"
#include "accessor_constructors_utility.h"
#include "accessor_constructors_image_utility.h"

namespace TEST_NAMESPACE {

/** tests the constructors for cl::sycl::accessor
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_all_dims(util::logger &log, cl::sycl::queue &queue) {
    image_accessor_dims<T, 1>::check(log, queue);
    image_accessor_dims<T, 2>::check(log, queue);
    image_accessor_dims<T, 3>::check(log, queue);
    image_array_accessor_dims<T, 1>::check(log, queue);
    image_array_accessor_dims<T, 2>::check(log, queue);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();
      if (!queue.get_device()
               .get_info<cl::sycl::info::device::image_support>()) {
        log.note("Device does not support images -- skipping check");
        return;
      }

      /** check accessor constructors for cl_int4
       */
      check_all_dims<cl::sycl::cl_int4>(log, queue);

      /** check accessor constructors for cl_uint4
       */
      check_all_dims<cl::sycl::cl_uint4>(log, queue);

      /** check accessor constructors for cl_float4
       */
      check_all_dims<cl::sycl::cl_float4>(log, queue);

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

}  // namespace TEST_NAMESPACE
