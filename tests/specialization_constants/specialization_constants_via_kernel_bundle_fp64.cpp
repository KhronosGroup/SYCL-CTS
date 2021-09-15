/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for specialization constants usage via kernel bundle for the
//  fp64 types
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "specialization_constants_via_kernel_bundle.h"

#define TEST_NAME specialization_constants_via_kernel_bundle_fp64

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace specialization_constants_via_kernel_bundle;

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
      auto queue = util::get_cts_object::queue();
      if (!queue.get_device().has(sycl::aspect::fp64)) {
        log.note(
            "Device does not support double precision floating point "
            "operations");
        return;
      }
#ifndef SYCL_CTS_FULL_CONFORMANCE
      check_all<double>{}(log, "double");
#else
      for_type_vectors_marray<check_all, double>(log, "double");
#endif
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    } catch (const std::exception &e) {
      std::string errorMsg =
          "an exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
