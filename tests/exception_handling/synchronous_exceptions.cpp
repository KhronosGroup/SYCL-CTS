/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME synchronous_exceptions

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/**
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    cts_selector selector;
    cl::sycl::queue q(selector);

    try {
      q.submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class TEST_NAME>([=]() {});
      });
    } catch (const cl::sycl::exception &e) {
      // Check methods
      cl::sycl::string_class sc = e.what();
      if (e.has_context()) {
        cl::sycl::context c = e.get_context();
      }
      cl::sycl::cl_int ci = e.get_cl_code();

      log_exception(log, e);
      FAIL(log, "An exception should not really have been thrown");
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // TEST_NAMESPACE
