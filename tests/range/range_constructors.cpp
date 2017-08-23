/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME range_constructors

namespace range_constructors__ {
using namespace sycl_cts;

/** test cl::sycl::range initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   *  @param log, test transcript logging class
   */
  virtual void run(util::logger &log) override {
    try {
      cl::sycl::range<1> range_one(1);
      cl::sycl::range<2> range_two(1, 2);
      cl::sycl::range<3> range_three(1, 2, 3);
      cl::sycl::range<1> range_four(range_one);
      cl::sycl::range<2> range_five(range_two);
      cl::sycl::range<3> range_six(range_three);
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace range_constructors__ */
