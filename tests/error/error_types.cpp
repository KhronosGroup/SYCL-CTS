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

#define TEST_NAME error_types

namespace error_types {
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

  /** execute this test
   */
  void run(util::logger &log) override {
#define CHECK_EXISTS(EXCEPTION_NAME)                       \
  if (!std::is_class<EXCEPTION_NAME>::value) {             \
    FAIL(log, "EXCEPTION_NAME is not defined as a class"); \
  }

    CHECK_EXISTS(cl::sycl::exception);
    CHECK_EXISTS(cl::sycl::runtime_error);
    CHECK_EXISTS(cl::sycl::kernel_error);
    CHECK_EXISTS(cl::sycl::accessor_error);
    CHECK_EXISTS(cl::sycl::nd_range_error);
    CHECK_EXISTS(cl::sycl::event_error);
    CHECK_EXISTS(cl::sycl::invalid_parameter_error);
    CHECK_EXISTS(cl::sycl::device_error);
    CHECK_EXISTS(cl::sycl::compile_program_error);
    CHECK_EXISTS(cl::sycl::link_program_error);
    CHECK_EXISTS(cl::sycl::invalid_object_error);
    CHECK_EXISTS(cl::sycl::memory_allocation_error);
    CHECK_EXISTS(cl::sycl::platform_error);
    CHECK_EXISTS(cl::sycl::profiling_error);
    CHECK_EXISTS(cl::sycl::feature_not_supported);
#undef CHECK_EXISTS

    /* Check that exception_list exists */
    if (!std::is_class<cl::sycl::exception_list>::value) {
      FAIL(log, "Exception list is not defined as a class");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace header_test_2__ */
