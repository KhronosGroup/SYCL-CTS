/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "atomic_constructors_common.h"

#include <string>

#define TEST_NAME atomic_constructors_32

namespace TEST_NAMESPACE {

using namespace atomic_constructors_common;
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
  void check_atomics_for_type(util::logger &log, cl::sycl::queue testQueue,
                              atomic64_bits_tag::no = atomic64_bits_tag::no{}) {
    /** Check atomic constructors for cl::sycl::access::target::global_buffer
     */
    check_atomics<T, cl::sycl::access::target::global_buffer>{}(log, testQueue);

    /** Check atomic constructors for cl::sycl::access::target::local
     */
    check_atomics<T, cl::sycl::access::target::local>{}(log, testQueue);
  }

  template <typename T>
  void check_atomics_for_type(util::logger &log, cl::sycl::queue testQueue,
                              atomic64_bits_tag::yes) {
    // Skip non-64bit checks for 64bit types
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

      auto skip_tag = atomic64_bits_tag::get<long>();
      check_atomics_for_type<long>(log, testQueue, skip_tag);
      check_atomics_for_type<unsigned long>(log, testQueue, skip_tag);

      testQueue.wait_and_throw();

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      auto errorMsg = std::string("a SYCL exception was caught: ") + e.what();
      FAIL(log, errorMsg);
    }
  }
};

// Construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
