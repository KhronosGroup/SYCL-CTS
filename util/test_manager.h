/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_TEST_MANAGER_H
#define __SYCLCTS_UTIL_TEST_MANAGER_H

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

  void dump_device_info();

  /** program lifetime hooks
   */
  void on_start();
  void on_exit();

 protected:
  bool m_willExecute;
  bool m_wimpyMode;
  bool m_infoDump;
  std::string m_infoDumpFile;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_TEST_MANAGER_H