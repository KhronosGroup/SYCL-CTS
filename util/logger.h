/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_LOGGER_H
#define __SYCLCTS_UTIL_LOGGER_H

#include "stl.h"
#include "test_base.h"

#include <string>
#include <string_view>
#include <type_traits>

namespace sycl_cts {
namespace util {

/** the logger class records all output during testing
 *  and so forms a transcript of an executed test
 */
class logger {
 public:
  /** test result values
   */
  enum result {
    epending = 0,
    epass,
    efail,
    eskip,
    efatal,
    etimeout,
  };

  /** constructor
   */
  logger();

  /** destructor
   */
  ~logger();

  /* emit a test preamble
   */
  void preamble(const struct test_base::info &testInfo);

  /** notify a test has failed
   */
  void fail(const std::string &reason, const int line);

  /** notify a test has been skipped
   */
  void skip(const std::string &reason = std::string());

  /** report fatal error and abort program
   */
  void fatal(const std::string &reason = std::string());

  /** output verbose information
   */
  void note(const std::string &str);

  /** output verbose information
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
      note(message);
    } else {
      // Using data value
      const std::string message{seed};
      note(message);
    }
#else
    static_cast<void>(seed);
#endif
  }

  /** beginning of a test
   */
  void test_start();

  /** end of a test
   */
  void test_end();

  /** send a progress update
   *
   *  sent as number of 'items' done of 'total'
   */
  void progress(int item, int total);

  /** return true if the log has been marked as fail
   */
  bool has_failed();

  /** return the test result as result enum
   */
  result get_result() const;

 protected:
  // unique log identifier
  int32_t m_logId;

  // test result
  result m_result;

  // disable copy constructors
  logger(const logger &);

};  // class logger

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_LOGGER_H