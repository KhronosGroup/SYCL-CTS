/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2021 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_LOGGER_H
#define __SYCLCTS_UTIL_LOGGER_H

#include "stl.h"

#include <string>
#include <type_traits>

namespace sycl_cts {
namespace util {

/**
 * Logging utility class for legacy test cases.
 * @deprecated Please use Catch2's INFO and WARN macros instead.
 */
class logger {
 public:
  logger() = default;

  /**
   * notify a test has failed
   * @deprecated Use the FAIL macro instead.
   */
  void fail(const std::string &reason, const int line);

  /**
   * output verbose information
   * @deprecated Use Catch2's WARN or INFO macros instead.
   */
  void note(const std::string &str);

  /**
   * output verbose information
   * @deprecated Use Catch2's WARN or INFO macros instead.
   */
  void note(const char *fmt, ...);

  /** @brief Output debug information
   *  @detail Provides output only if the SYCL_CTS_VERBOSE_LOG macro was defined
   *  during compilation.
   *  Lambda can be used to make the message construction as lazy as possible.
   *  Usage examples:
   *
   *      log.debug([&]{
   *        std::string message{"Running test for "};
   *        message += typeName;
   *        return message;
   *      });
   *      log.debug("Message");
   *      log.debug("String view instance"sv);
   *
   *  @param seed Either lambda returning the message to log or the message
   *              itself
   */
  template <typename seedT>
  void debug(const seedT &seed) {
#ifdef SYCL_CTS_VERBOSE_LOG
    if constexpr (std::is_invocable_v<seedT>) {
      // Using factory method
      const std::string message{seed()};
      log_debug(message);
    } else {
      // Using data value
      const std::string message{seed};
      log_debug(message);
    }
#else
    static_cast<void>(seed);
#endif
  }

  // This is a hack to enable both old and new style FAIL macros, i.e.
  // FAIL(log, "my message") // old style
  // FAIL("my value = " << 123) // new Catch2 style
  friend std::ostream &operator<<(std::ostream &os, const logger &m) {
    return os;
  }

 protected:
  // disable copy constructors
  logger(const logger &);

  void log_debug(const std::string &str);

};  // class logger

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_LOGGER_H