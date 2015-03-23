/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_info

namespace platform_info__ {
using namespace sycl_cts;

/** tests the info for cl::sycl::platform
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
    try {
      cl::sycl::platform platform;

      /** check types
      */
      using platformInfo = cl::sycl::info::platform;

      /** initialize return values
      */
      cl::sycl::string_class stringClassRet;

      /** check platform info parameters
      */
      stringClassRet = platform.get_info<cl::sycl::info::platform::vendor>();
      stringClassRet = platform.get_info<cl::sycl::info::platform::name>();
      stringClassRet = platform.get_info<cl::sycl::info::platform::version>();
      stringClassRet = platform.get_info<cl::sycl::info::platform::profile>();
      stringClassRet =
          platform.get_info<cl::sycl::info::platform::extensions>();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace platform_info__ */
