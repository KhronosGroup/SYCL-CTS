/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide exceptions to handle the test logic
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_EXCEPTIONS_H
#define __SYCLCTS_UTIL_EXCEPTIONS_H

#include "../tests/common/macros.h"
#include "logger.h"

#include <stdexcept>
#include <utility>

namespace sycl_cts {
namespace util {

/** @brief Exception to be thrown if the single check should be skipped, but the
 *         test itself should continue to run
 *  @details This exception is designed to be handled by the test code itself
 *           without propagation to the test_base
 */
class skip_check : public std::runtime_error {
 public:
  template <typename argT>
  skip_check(argT &&arg) : std::runtime_error(std::forward<argT>(arg)) {}
};

/** @brief Exception to be thrown if the single check should be reported as the
 *         failed one, but the test itself should continue to run
 *  @details This exception is designed to be handled by the test code itself
 *           without propagation to the test_base
 */
class fail_check : public std::runtime_error {
 public:
  template <typename argT>
  fail_check(argT &&arg) : std::runtime_error(std::forward<argT>(arg)) {}
};

/** @brief Provides in-place exception handling to catch expected exceptions
 *         and do not stop testing process
 */
template <typename F>
static void run_check(logger &log, const std::string &checkName, F functor) {
  try {
    functor();
  } catch (const skip_check &skip_e) {
    log.note(checkName + " " + skip_e.what() + " (test skipped)");
  } catch (const fail_check &fail_e) {
    FAIL(log, checkName + " " + fail_e.what());
  } catch (...) {
    throw;
  }
}

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_EXCEPTIONS_H
