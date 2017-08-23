/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

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
      /** check info::platform
       */
      check_enum_class_value(cl::sycl::info::platform::profile);
      check_enum_class_value(cl::sycl::info::platform::version);
      check_enum_class_value(cl::sycl::info::platform::name);
      check_enum_class_value(cl::sycl::info::platform::vendor);
      check_enum_class_value(cl::sycl::info::platform::extensions);

      /** check get_info parameters
       */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        check_get_info_param<cl::sycl::info::platform, cl::sycl::string_class,
                             cl::sycl::info::platform::profile>(log, plt);
        check_get_info_param<cl::sycl::info::platform, cl::sycl::string_class,
                             cl::sycl::info::platform::version>(log, plt);
        check_get_info_param<cl::sycl::info::platform, cl::sycl::string_class,
                             cl::sycl::info::platform::name>(log, plt);
        check_get_info_param<cl::sycl::info::platform, cl::sycl::string_class,
                             cl::sycl::info::platform::vendor>(log, plt);
        check_get_info_param<cl::sycl::info::platform,
                             cl::sycl::vector_class<cl::sycl::string_class>,
                             cl::sycl::info::platform::extensions>(log, plt);
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

} /* namespace platform_info__ */
