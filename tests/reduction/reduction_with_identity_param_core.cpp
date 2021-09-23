/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for arithmetic
//  scalar types.
//
*******************************************************************************/

#include "reduction_with_identity_param.h"

#define TEST_NAME reduction_with_identity_param_core

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
    try {
      auto queue = util::get_cts_object::queue();

      for_all_types<reduction_with_identity::run_test_for_type>(
          reduction_common::scalar_types, queue, log);
    } catch (const sycl::exception& e) {
      log_exception(log, e);
      auto errorMsg = "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
