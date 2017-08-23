/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#pragma once

#include "singleton.h"

namespace sycl_cts {
namespace util {

/** manage the overall state of the test executable
 *
 */
class test_manager : public singleton<test_manager> {
 public:
  /** constructor
   */
  test_manager();

  /** parse the command line options
   */
  bool parse(const int argc, const char **args);

  /** run the tests themselves
   */
  bool run();

  /** print command line usage information to the screen
   */
  void print_usage();

  /**
   */
  bool will_execute() const;

  /**
   */
  bool wimpy_mode_enabled() const;

  /** program lifetime hooks
   */
  void on_start();
  void on_exit();

 protected:
  bool m_willExecute;
  bool m_wimpyMode;
};

}  // namespace util
}  // namespace sycl_cts
