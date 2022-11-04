/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification to atomic access for USM allocations with
//  double-precision as their underlying type.
//
*******************************************************************************/

#include "usm_atomic_access.h"

#define TEST_NAME usm_atomic_access_atomic64_fp64

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
    {
      auto queue{util::get_cts_object::queue()};
      if (!queue.get_device().has(sycl::aspect::fp64)) {
        WARN(
            "Device does not support double precision floating point "
            "operations. Skipping the test case.");
        return;
      }
      for_all_types<usm_atomic_access::run_all_tests>(
          usm_atomic_access::get_fp64_type(), queue, log,
          usm_atomic_access::with_atomic64);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
