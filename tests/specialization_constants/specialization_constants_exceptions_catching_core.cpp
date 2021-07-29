/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for specialization constants expected exceptions catching
//
*******************************************************************************/

#include "../common/common.h"

#include "specialization_constants_exceptions_catching_common.h"

#define TEST_NAME specialization_constants_exceptions_catching_core

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test specialization constants expected exceptions catching
 */
class TEST_NAME : public util::test_base {
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
#ifndef SYCL_CTS_FULL_CONFORMANCE
      for_all_types<check_spec_constant_except_catch_for_type>(
          get_spec_const::testing_types::types, log);
#else
      for_all_types_vectors_marray<check_spec_constant_except_catch_for_type>(
          get_spec_const::testing_types::types, log);
#endif
      for_all_types<check_spec_constant_except_catch_for_type>(
          get_spec_const::testing_types::composite_types, log);
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
