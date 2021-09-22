/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides test_result_checker memeber functions definitions
//
*******************************************************************************/
#include "exceptions_sycl_category_common.h"

using namespace exceptions_sycl_category_common;

void test_result_checker::compare_sycl(
    const std::error_category &error_category, sycl::errc err_c) {
  m_tests_results[err_c] =
      std::error_condition(static_cast<int>(err_c), sycl::sycl_category()) ==
      error_category.default_error_condition(static_cast<int>(err_c));
}

void test_result_checker::check_results(const std::string_view &error_msg,
                                        sycl_cts::util::logger &log) const {
  for (const auto &[err_code, test_result] : m_tests_results) {
    if (!test_result) {
      FAIL(log, std::string(error_msg) +
                    " std::error_condition is not equal to "
                    "error_category::default_error_condition for sycl::errc::" +
                    errc_to_string(err_code));
    }
  }
}
