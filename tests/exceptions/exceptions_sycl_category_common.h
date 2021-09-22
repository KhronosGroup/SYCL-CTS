/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common code for exceptions sycl category tests
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_EXCEPTIONS_SYCL_CATEGORY_COMMON_H
#define __SYCL_CTS_TEST_EXCEPTIONS_SYCL_CATEGORY_COMMON_H

#include "../common/common.h"
#include "exceptions.h"
#include <map>

namespace exceptions_sycl_category_common {

/** @brief Class for unite functions that used in test results check
 */
class test_result_checker {
  /** @brief Save the compare result between std::error_condition and
   *         error_category::default_error_condition in static dictionary
   *  @param error_category std::error_category class object
   *  @param err_c Error code from sycl::errc enumeration
   */
  void compare_sycl(const std::error_category &error_category,
                    sycl::errc err_c);

  std::map<sycl::errc, bool> m_tests_results{};

 public:
  test_result_checker(const std::error_category &error_category) {
    const auto all_err_codes {get_err_codes()};
    for (auto &err_c : all_err_codes) {
      test_result_checker::compare_sycl(error_category, err_c);
    }
  }

  /** @brief Checking std::map with comparing result for one of two test cases
   *  @param error_message Error message for logging
   *  @param log sycl_cts::util::logger class object
   */
  void check_results(const std::string_view &error_msg,
                     sycl_cts::util::logger &log) const;
};

}  // namespace exceptions_sycl_category_common

#endif  // __SYCL_CTS_TEST_EXCEPTIONS_SYCL_CATEGORY_COMMON_H
