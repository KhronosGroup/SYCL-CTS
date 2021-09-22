/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::errc enumeration's fields
//
*******************************************************************************/

#include "exceptions.h"
#include <set>

#define TEST_NAME exceptions_error_code

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** @brief Provide verification that sycl::errc enumeration fields have unique
 * values
 *  @param err_c Error code from sycl::errc enumeration
 *  @param log sycl_cts::util::logger class object
 */
void check_unique_enum_values(sycl::errc err_c, util::logger &log) {
  static std::set<int> error_code_values{};
  int integer_representation = static_cast<int>(err_c);
  if (error_code_values.find(integer_representation) !=
      error_code_values.end()) {
    FAIL(log, errc_to_string(err_c) +
                 " error code have same value that another enumeration field");
  }
  error_code_values.insert(integer_representation);
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
      const auto all_err_codes {get_err_codes()};
      for (auto &err_c : all_err_codes) {
        check_unique_enum_values(err_c, log);
      }
      if (!std::is_error_code_enum<sycl::errc>::value) {
        FAIL(log, "sycl::errc is not a error code enumeration");
      }
      if (std::is_error_condition_enum<sycl::errc>::value) {
        FAIL(log, "sycl::errc is a error condition enumeration");
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
