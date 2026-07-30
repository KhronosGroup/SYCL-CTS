/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2018-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "atomic_constructors_common.h"

#include <climits>
#include <string>

#define TEST_NAME atomic_constructors_64

namespace TEST_NAMESPACE {

using namespace atomic_constructors_common;
using namespace sycl_cts;

/** Check the api for sycl::atomic
 */
class TEST_NAME : public util::test_base {
 public:
  /** Return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_atomics_for_type(util::logger &log, sycl::queue testQueue) {
    /** Check atomic constructors for sycl::target::device
     */
    check_atomics<T, sycl::target::device>{}(log, testQueue);

    /** Check atomic constructors for sycl::target::local
     */
    check_atomics<T, sycl::target::local>{}(log, testQueue);
  }

  /** Execute the test
   */
  virtual void run(util::logger &log) override {
    {
      auto testQueue = util::get_cts_object::queue();
      auto testDevice = testQueue.get_device();

      /** Check atomics for supported types
       */
      if (testDevice.has(sycl::aspect::atomic64)) {
        if constexpr (sizeof(long) * CHAR_BIT == 64 /*bits*/) {
          check_atomics_for_type<long>(log, testQueue);
          check_atomics_for_type<unsigned long>(log, testQueue);
        }
        check_atomics_for_type<long long>(log, testQueue);
        check_atomics_for_type<unsigned long long>(log, testQueue);
      }

      testQueue.wait_and_throw();
    }
  }
};

// Construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
