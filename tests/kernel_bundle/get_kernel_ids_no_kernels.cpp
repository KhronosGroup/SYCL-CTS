/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel_id::get_kernel_ids() for zero kernels in
//  application.
//
//  IMPORTANT: This file should not be merged with another *.cpp file because we
//  must have a strictly defined kernels in the application because tested
//  function uses all available kernels and we can't chose specific kernel.
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_id.h"
#include "kernel_bundle.h"

#define TEST_NAME get_kernel_ids_no_kernels

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::get_kernel_id;
using namespace sycl_cts::tests::kernel_bundle;

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
    /** Call sycl::get_kernel_ids() without any kernels in application
     */
    run_verification<by_handler>(log, [&](sycl::handler &cgh) {
      auto k_ids = sycl::get_kernel_ids();
      if (k_ids.size() > 0) {
        FAIL(log, "One or more kernel_ids returned (" +
                      std::to_string(k_ids.size()) +
                      "). Return vector should not"
                      " include built-in kernel ids from any device");
      }
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
