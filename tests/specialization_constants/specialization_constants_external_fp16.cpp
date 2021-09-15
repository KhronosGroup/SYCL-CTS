/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for specialization constants with SYCL_EXTERNAL function
//  for sycl::half
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"

#define TEST_FP16

#include "specialization_constants_external.h"

#define TEST_NAME specialization_constants_external_fp16

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test specialization constants for sycl::half
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
#ifndef SYCL_EXTERNAL
    log.note("SYCL_EXTERNAL is not defined");
#else
    using namespace specialization_constants_external;
    try {
      auto queue = util::get_cts_object::queue();
      if (!queue.get_device().has(sycl::aspect::fp16)) {
        log.note(
            "Device does not support half precision floating point "
            "operations");
        return;
      }
#ifndef SYCL_CTS_FULL_CONFORMANCE
      check_specialization_constants_external<sycl::half> fp16_test{};
      fp16_test(log, "sycl::half");
#else
      for_type_vectors_marray<check_specialization_constants_external,
                              sycl::half>(log, "sycl::half");
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
#endif  // SYCL_EXTERNAL
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
