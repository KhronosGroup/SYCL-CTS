/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Generic assertions we can use for every test. The intent is to have:
//   - some generic assertions we can use for any lambda/functor with specific
//     test case logic
//   - some interface for test case specific log messages within such generic
//     assertions
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_ASSERTIONS_H
#define __SYCLCTS_TESTS_COMMON_ASSERTIONS_H

#include "../../util/logger.h"
#include "../common/macros.h"
#include <string>

/** @brief Interface for any test case description class to use for logs within
 *         generic assertions
 *  @details Test developer is free to provide any child with any parameters and
 *           log message for specific subset of tests.
 */
class ITestCaseDescription {
 public:
  virtual ~ITestCaseDescription() = default;

  /** @brief Method to be called during notice/failure message construction
   */
  virtual std::string to_string() const = 0;
};

/** @brief Assertion to verify the `action` doesn't throw any exception
 *  @details Use case can be as follows:
 *               expect_not_throws(log, desc_t("prefetch"), [&]{
 *                 cgh.prefetch(ptr, numBytes);
 *               });
 */
template <typename ActionT, typename ... ActionArgsT>
void expect_not_throws(sycl_cts::util::logger &log,
                       const ITestCaseDescription& description, ActionT action,
                       ActionArgsT&& ... args) {
  try {
    action(std::forward<ActionArgsT>(args)...);
  } catch (const sycl::exception& e) {
      std::string message = "Unexpected exception thrown for ";
      message += description.to_string();

      // Print detailed description specific for the test case
      FAIL(log, message);
      // Re-trow exception; it's OK as we generally don't know how to handle it
      throw;
  }
}

/** @brief Assertion to verify the `action` throws exactly the exception given
 *  @details Use case can be as follows:
 *               expect_throws<sycl::errc::invalid>(
 *                 log, desc_t("with context, devices"),
 *                 [&]{sycl::get_kernel_bundle<State>(context, zero_device);});
 */
template <sycl::errc expected, typename ActionT, typename ... ActionArgsT>
void expect_throws(sycl_cts::util::logger &log,
                  const ITestCaseDescription& description, ActionT action,
                  ActionArgsT&& ... args) {
  bool success = false;
  try {
    action(std::forward<ActionArgsT>(args)...);
  } catch (const sycl::exception& e) {
    // Verify exception is as expected
    success = (e.category() == sycl::sycl_category()) && (e.code() == expected);
    if (!success) {
      std::string message = "Unexpected exception thrown for ";
      message += description.to_string();

      FAIL(log, message);
      throw;
    }
  }
  if (!success) {
    std::string message = "No expected exception thrown for ";
    message += description.to_string();

    FAIL(log, message);
  }
}

#endif  // __SYCLCTS_TESTS_COMMON_ASSERTIONS_H
