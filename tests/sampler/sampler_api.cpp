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

#define TEST_NAME sampler_apis

namespace sampler_api__ {
using namespace sycl_cts;

/** tests the API for cl::sycl::sampler
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
      // Ensure all addressing_mode values defined
      check_enum_class_value(cl::sycl::addressing_mode::mirrored_repeat);
      check_enum_class_value(cl::sycl::addressing_mode::repeat);
      check_enum_class_value(cl::sycl::addressing_mode::clamp_to_edge);
      check_enum_class_value(cl::sycl::addressing_mode::clamp);
      check_enum_class_value(cl::sycl::addressing_mode::none);
      check_enum_underlying_type<cl::sycl::addressing_mode, unsigned int>(log);

      // Ensure all filtering_mode values defined
      check_enum_class_value(cl::sycl::filtering_mode::nearest);
      check_enum_class_value(cl::sycl::filtering_mode::linear);
      check_enum_underlying_type<cl::sycl::filtering_mode, unsigned int>(log);

      // Ensure all coordinate_normalization_mode values defined
      check_enum_class_value(
          cl::sycl::coordinate_normalization_mode::normalized);
      check_enum_class_value(
          cl::sycl::coordinate_normalization_mode::unnormalized);
      check_enum_underlying_type<cl::sycl::coordinate_normalization_mode,
                                 unsigned int>(log);

      cl::sycl::sampler sampler(
          cl::sycl::coordinate_normalization_mode::unnormalized,
          cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);

      /** check get_addressing_mode() method
      */
      auto addressingMode = sampler.get_addressing_mode();
      check_return_type<cl::sycl::addressing_mode>(log, addressingMode,
                                                   "get_addressing_mode()");

      /** check get_filtering_mode() method
      */
      auto filterMode = sampler.get_filtering_mode();
      check_return_type<cl::sycl::filtering_mode>(log, filterMode,
                                                  "get_filtering_mode()");

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

} /* namespace sampler_api__ */
