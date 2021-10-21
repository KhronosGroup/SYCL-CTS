/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2021 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_TEST_BASE_H
#define __SYCLCTS_UTIL_TEST_BASE_H

#include <string>

#include "logger.h"

// conformance test suite namespace
namespace sycl_cts {
namespace util {

/**
 * Base class for legacy test cases.
 * @deprecated Use Catch2's TEST_CASE macro instead.
 */
class test_base {
 public:
  /** encapsulate information about a test
   */
  struct info {
    std::string m_name;
    std::string m_file;
    std::string m_buildDate;
    std::string m_buildTime;
  };

  /** virtual destructor
   */
  virtual ~test_base() {
    /* call cleanup to ensure internals are released */
    cleanup();
  }

  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  virtual void get_info(info &out) const = 0;

  // Some legacy tests change visibility of get_info to non-public
  void get_info_legacy(info &out) { get_info(out); }

  /** called before this test is executed
   *  @param log for emitting test notes and results
   */
  virtual bool setup(class logger &) { return true; }

  /** member function that will be overridden in test file
   *  @param log for emitting test notes and results
   */
  virtual void run(class logger &log) = 0;

  /** overridden member function with try-catch block
   *  @param log for emitting test notes and results
   */
  void run_test(class logger &log);

  void run_legacy() {
    logger log{};
    run_test(log);
  }

  /** called after this test has executed
   *  @param log for emitting test notes and results
   */
  virtual void cleanup() {
    // empty
  }

};  // class test_base

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_TEST_BASE_H