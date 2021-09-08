/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::make_error_code function
//
*******************************************************************************/

#include "exceptions.h"

#define TEST_NAME exceptions_make_error_code

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** @brief Provide verification for sycl::make_error_code function
 *  @param err_c Error code from sycl::errc enumeration
 *  @param log sycl_cts::util::logger class object
 */
void check_sycl_working(sycl::errc err_c, util::logger &log) {
  auto make_err_c_result{sycl::make_error_code(err_c)};

  if (!noexcept(sycl::make_error_code(err_c))) {
    FAIL(log, "sycl::make_error_code function are not marked as \"noexcept\"");
  }
  CHECK_TYPE(log, make_err_c_result, std::error_code());
  if (make_err_c_result != err_c) {
    FAIL(
        log,
        "sycl::make_error_code function's error code are not equal to provided "
        "error code from sycl::errc enumeration");
  }
}

/** @brief Provide verification for same work std::error_code and
 *         sycl::make_error_code
 *  @param err_c Error code from sycl::errc enumeration
 *  @param log sycl_cts::util::logger class object
 */
void compare_sycl_and_std_working(sycl::errc err_c, util::logger &log) {
  auto err_c_result{
      std::error_code(static_cast<int>(err_c), sycl::sycl_category())};
  auto make_err_c_result{sycl::make_error_code(err_c)};

  if (err_c_result.value() != make_err_c_result.value()) {
    FAIL(log,
         "error code value that received from std::error_code not equal to "
         "value that received from sycl::make_error_code ");
  }
  if (err_c_result.message().empty()) {
    FAIL(log, "error message from std::error_code are empty");
  }
  if (make_err_c_result.message().empty()) {
    FAIL(log, "error message from sycl::make_error_code are empty");
  }
  if (err_c_result.message() != make_err_c_result.message()) {
    FAIL(log,
         "error message from std::error_code not equal to error message from "
         "sycl::make_error_code");
  }
  if (err_c_result.default_error_condition() !=
      std::error_condition(static_cast<int>(err_c), sycl::sycl_category())) {
    FAIL(log,
         "default error condition that received from std::error_code not equal "
         "to error condition that received from std::error_condition");
  }
  if (make_err_c_result.default_error_condition() !=
      std::error_condition(static_cast<int>(err_c), sycl::sycl_category())) {
    FAIL(log,
         "default error condition that received from sycl::make_error_code not "
         "equal to error condition that received from std::error_condition");
  }
  if (err_c_result.category() != sycl::sycl_category()) {
    FAIL(log,
         "error category that received from std::error_code not equal to "
         "sycl::sycl_category");
  }
  if (make_err_c_result.category() != sycl::sycl_category()) {
    FAIL(log,
         "error category that received from sycl::make_error_code not equal to "
         "sycl::sycl_category");
  }
}

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
    try {
      for (auto &err_c : exceptions::all_err_codes) {
        compare_sycl_and_std_working(err_c, log);
        check_sycl_working(err_c, log);
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg{"a SYCL exception was caught: " +
                           std::string(e.what())};
      FAIL(log, errorMsg);
    } catch (const std::exception &e) {
      std::string errorMsg{"an exception was caught: " + std::string(e.what())};
      FAIL(log, errorMsg);
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
