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

#include "../common/common.h"
#include "atomic_constructors_common.h"
#include <sstream>

#define TEST_NAME atomic_constructors_32

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** Check the api for cl::sycl::atomic
*/
class TEST_NAME : public util::test_base {
 public:
  /** Return information about this test
  */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_atomics_for_type(util::logger &log, cl::sycl::queue testQueue) {
    /** Check atomic constructors for cl::sycl::access::target::global_buffer
    */
    check_atomics<T, cl::sycl::access::target::global_buffer>{}(log, testQueue);

    /** Check atomic constructors for cl::sycl::access::target::local
    */
    check_atomics<T, cl::sycl::access::target::local>{}(log, testQueue);
  }

  /** Execute the test
  */
  virtual void run(util::logger &log) override {
    try {
      auto testQueue = util::get_cts_object::queue();

      /** Check atomics for supported types
      */
      check_atomics_for_type<int>(log, testQueue);
      check_atomics_for_type<unsigned int>(log, testQueue);
      check_atomics_for_type<float>(log, testQueue);

      testQueue.wait_and_throw();

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// Construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
