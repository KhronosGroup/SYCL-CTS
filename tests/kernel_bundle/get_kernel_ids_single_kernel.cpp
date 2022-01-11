/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel_id::get_kernel_ids() for single kernel in
//  application.
//
//  IMPORTANT: This file should not be merged with another *.cpp file because we
//  must have a strictly defined kernels in the application because tested
//  function uses all available kernels and we can't chose specific kernel.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/common_by_reference.h"
#include "get_kernel_id.h"
#include "kernel_bundle.h"

#define TEST_NAME get_kernel_ids_single_kernel

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
    /** Call sycl::get_kernel_id() for single kernel in application
     */
    run_verification<by_handler>(log, [&](sycl::handler &cgh) {
      using kernel_name = class get_kernel_ids_single;
      auto k_ids = sycl::get_kernel_ids();
      if (k_ids.size() == 0) {
        FAIL(log,
             "Empty kernel_ids vector returned. Vector should include"
             "exactly one kernel_id");
      } else if (k_ids.size() > 1) {
        FAIL(log, "More than one kernel_ids in returned vector (" +
                      std::to_string(k_ids.size()) +
                      "). Vector should"
                      " include exactly one kernel_id");
      }
      cgh.single_task<kernel_name>([=]() {});
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
