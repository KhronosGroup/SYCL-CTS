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

#define TEST_NAME context_info

namespace contect_info__ {
using namespace sycl_cts;

/** tests the info for sycl::context
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
      auto context = util::get_cts_object::context();

      /** check get_info for info::context::reference_count
       */
      {
        auto ref_count =
            context.get_info<sycl::info::context::reference_count>();
        check_return_type<cl_uint>(
            log, ref_count,
            "get_info<sycl::info::context::reference_count>()");
        TEST_TYPE_TRAIT(context, reference_count, context);
      }

      /** check get_info for info::context::platform
       */
      {
        auto platform = context.get_info<sycl::info::context::platform>();
        check_return_type<sycl::platform>(
            log, platform, "get_info<sycl::info::context::platform>()");
        TEST_TYPE_TRAIT(context, platform, context);
      }

      /** check get_info for info::context::devices
       */
      {
        auto devs = context.get_info<sycl::info::context::devices>();
        check_return_type<std::vector<sycl::device>>(
            log, devs, "get_info<sycl::info::context::devices>()");
        TEST_TYPE_TRAIT(context, devices, context);
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace context_info__ */
