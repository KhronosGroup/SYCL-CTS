/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Factory methods for objects created once per translation unit
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_ONCE_PER_UNIT_H
#define __SYCLCTS_TESTS_COMMON_ONCE_PER_UNIT_H

#include "../../util/logger.h"
#include "../common/get_cts_object.h"

namespace detail {
/**
 * @brief Helper class for once_per_unit::log() function
 */
struct log_notice {
  log_notice(sycl_cts::util::logger &log, const std::string &message) {
    log.note(message);
  }
};

}  // namespace detail

namespace {
/**
 * All symbols have internal linkage here;
 * special attention to the ODR rules should be made
 */
namespace once_per_unit {
/**
 * @brief Factory method; provides unique queue instance per compilation unit
 */
inline sycl::queue &get_queue() {
  static auto q = sycl_cts::util::get_cts_object::queue();
  return q;
}

/**
 * @brief Provide possibility to log message once per translation unit
 */
static void log(sycl_cts::util::logger &log, const std::string &message) {
  static const detail::log_notice log_just_once(log, message);
}
}  // namespace once_per_unit
}  // namespace

#endif  // __SYCLCTS_TESTS_COMMON_ONCE_PER_UNIT_H
