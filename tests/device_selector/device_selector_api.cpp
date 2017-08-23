/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_selector_api

namespace device_selector_api__ {
using namespace sycl_cts;

/** tests the api for cl::sycl::device_selector
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      /** check select_device() method
       */
      {
        cts_selector selector;
        auto selected_device = selector.select_device();
        check_return_type<cl::sycl::device>(log, selected_device,
                                            "select_device()");
      }

      /** check ()(device) operator
       */
      {
        cl::sycl::device device;
        cts_selector selector;
        auto score = selector(device);
        check_return_type<int>(log, score, "selector(cl::sycl::device)");
      }

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_selector_api__ */
