/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for interaction reductions with sycl::half variable type
//  without identity param.
//
*******************************************************************************/

#include "reduction_without_identity_param_common.h"

#define TEST_NAME reduction_without_identity_param_fp16

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace reduction_without_identity;
using namespace reduction_common;

/** Test instance
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp16)) {
      log.note(
          "Device does not support half precision floating point operations");
      return;
    }

    run_tests_for_all_functors<sycl::half, run_test_without_property>()(
        range, queue, log, "sycl::half");
    run_tests_for_all_functors<sycl::half, run_test_with_property>()(
        nd_range, queue, log, "sycl::half");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
