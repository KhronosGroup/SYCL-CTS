/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_selector_api

namespace device_selector_api__ {
using namespace sycl_cts;

/** tests the api for sycl::device_selector
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
    {
      /** check select_device() method
       */
      {
        cts_selector selector;
        auto selected_device = selector.select_device();
        check_return_type<sycl::device>(log, selected_device,
                                            "select_device()");
      }

      /** check ()(device) operator
       */
      {
        sycl::device device;
        cts_selector selector;
        auto score = selector(device);
        check_return_type<int>(log, score, "selector(sycl::device)");
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_selector_api__ */
