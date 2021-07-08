/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME context_api

namespace context_api__ {
using namespace sycl_cts;

/** tests the api for sycl::context
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
      auto context = util::get_cts_object::context();

      /** check is_host() method
       */
      {
        auto isHost = context.is_host();
        check_return_type<bool>(log, isHost, "is_host()");
      }

      /** check get_devices() method
       */
      {
        auto deviceList = context.get_devices();
        check_return_type<sycl::vector_class<sycl::device>>(
            log, deviceList, "get_devices()");
      }

      /** check get_platform() method
       */
      {
        auto platform = context.get_platform();
        check_return_type<sycl::platform>(log, platform, "get_platform()");
      }

    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace context_api */
