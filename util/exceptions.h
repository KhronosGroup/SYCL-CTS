/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide exceptions to handle the test logic
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_EXCEPTIONS_H
#define __SYCLCTS_UTIL_EXCEPTIONS_H

#include <stdexcept>
#include <utility>

namespace sycl_cts {
namespace util {

/** @brief Exception to be thrown if the single check should be skipped, but the
 *         test itself should continue to run
 *  @details This exception is designed to be handled by the test code itself
 */
class skip_check : public std::runtime_error {
 public:
  template <typename argT>
  skip_check(argT&& arg) : std::runtime_error(std::forward<argT>(arg)) {}
};

/** @brief Exception to be thrown if the single check should be reported as the
 *         failed one, but the test itself should continue to run
 *  @details This exception is designed to be handled by the test code itself
 */
class fail_check : public std::runtime_error {
 public:
  template <typename argT>
  fail_check(argT&& arg) : std::runtime_error(std::forward<argT>(arg)) {}
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_EXCEPTIONS_H
