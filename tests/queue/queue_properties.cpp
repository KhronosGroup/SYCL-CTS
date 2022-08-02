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

#define TEST_NAME queue_properties

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** tests the properties for sycl::queue
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  virtual void run(util::logger &log) override {
    {
      auto queue = util::get_cts_object::queue();

      /** check property::queue::enable_profiling
      */
      {
        sycl::queue queue(
            util::get_cts_object::device(),
            sycl::property_list{
                sycl::property::queue::enable_profiling()});

        if (!queue
                 .has_property<sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with enable_profiling property was not constructed "
               "correctly");
        }

        auto prop =
            queue.get_property<sycl::property::queue::enable_profiling>();
        check_return_type<sycl::property::queue::enable_profiling>(
            log, prop,
            "sycl::queue::has_property<sycl::property::queue::"
            "enable_profiling>()");
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
