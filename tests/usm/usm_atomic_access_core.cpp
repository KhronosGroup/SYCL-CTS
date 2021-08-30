/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification to atomic access for USM allocations that underlying
//  type size lower than 64 byte.
//
*******************************************************************************/

#include "usm_atomic_access.h"

#define TEST_NAME usm_atomic_access_core

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** Test instance
 */
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
    try {
      auto queue{util::get_cts_object::queue()};

      for_all_types<usm_atomic_access::run_all_tests,
                    usm_atomic_access::without_atomic64>(
          usm_atomic_access::get_scalar_types(), queue, log);
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      auto errorMsg = "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    } catch (const std::exception &e) {
      auto errorMsg = "a std exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
