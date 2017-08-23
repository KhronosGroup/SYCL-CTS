/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#pragma once

#include "stl.h"
#include "test_base.h"

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
