/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_LOGGER_H
#define __SYCLCTS_UTIL_LOGGER_H

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

#endif  // __SYCLCTS_UTIL_LOGGER_H