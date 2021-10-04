/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for sycl::half.
//
*******************************************************************************/

#include "reduction_with_identity_param.h"

#define TEST_NAME reduction_with_identity_param_fp16

namespace TEST_NAMESPACE {
using namespace sycl_cts;

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
    {
      auto queue = util::get_cts_object::queue();
      if (!queue.get_device().has(sycl::aspect::fp16)) {
        log.note(
            "Device does not support half precision floating point operations");
        return;
      }
      reduction_with_identity::run_test_for_type<sycl::half>()(queue, log,
                                                               "sycl::half");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
