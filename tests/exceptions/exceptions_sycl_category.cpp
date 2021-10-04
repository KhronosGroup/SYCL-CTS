/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::sycl_category function
//
*******************************************************************************/

#include "exceptions.h"
#include "exceptions_sycl_category_common.h"

#define TEST_NAME exceptions_sycl_category

namespace TEST_NAMESPACE {

using namespace sycl_cts;
using namespace exceptions_sycl_category_common;

/** Test instance
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
    {
      const auto &error_category_local_usage{sycl::sycl_category()};
      const test_result_checker test_result_checker_local_usage(
          error_category_local_usage);
      test_result_checker_local_usage.check_results(
          "in locally using std::error_category,", log);
      if (!noexcept(sycl::sycl_category())) {
        FAIL(log,
             "sycl::sycl_category function are not marked as \"noexcept\"");
      }
      auto err_c{sycl::errc::accessor};
      if (error_category_local_usage.message(static_cast<int>(err_c)).empty()) {
        FAIL(log, "error category message are empty");
      }
      if (strcmp(error_category_local_usage.name(), "sycl") != 0) {
        FAIL(log, "sycl::sycl_category name is not equal to \"sycl\"");
      }
      if (!std::is_same<decltype(sycl::sycl_category()),
                        const std::error_category &>::value) {
        FAIL(log,
             "sycl::sycl_category function's return type are not equal to "
             "const std::error_category");
      }
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
