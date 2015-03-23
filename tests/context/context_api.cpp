/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME context_api

namespace context_api__ {
using namespace sycl_cts;

/** tests the api for cl::sycl::context
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
      cts_selector selector;
      cl::sycl::context context(selector);

      /** check is_host() method
      */
      auto isHost = context.is_host();
      if (typeid(isHost) != typeid(bool)) {
        FAIL(log, "is_host() does not return bool");
      }

      /** check get_devices() method
      */
      auto deviceList = context.get_devices();
      if (typeid(isHost) != typeid(cl::sycl::vector_class<cl::sycl::device>)) {
        FAIL(log, "get_devices() does not return vector_class<device>");
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace context_api */
