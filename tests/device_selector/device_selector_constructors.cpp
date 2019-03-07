/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_selector_constructors

namespace device_selector_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::device_selector
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
    try {
      /** check default constructor and destructor
      */
      { cts_selector selector; }

      /** check copy constructor
      */
      {
        cts_selector selectorA;
        cts_selector selectorB(selectorA);
      }

      /** check assignment operator
      */
      {
        cts_selector selectorA;
        cts_selector selectorB = selectorA;
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_selector_constructors__ */
