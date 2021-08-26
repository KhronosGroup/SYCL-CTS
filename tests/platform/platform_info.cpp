/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_info

namespace platform_info__ {
using namespace sycl_cts;

/** tests the info for sycl::platform
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    /** check info::platform
     */
    check_enum_class_value(sycl::info::platform::profile);
    check_enum_class_value(sycl::info::platform::version);
    check_enum_class_value(sycl::info::platform::name);
    check_enum_class_value(sycl::info::platform::vendor);
    check_enum_class_value(sycl::info::platform::extensions);

    /** check get_info parameters
     */
    {
      cts_selector selector;
      auto plt = util::get_cts_object::platform(selector);
      check_get_info_param<sycl::info::platform, std::string,
                           sycl::info::platform::profile>(log, plt);
      check_get_info_param<sycl::info::platform, std::string,
                           sycl::info::platform::version>(log, plt);
      check_get_info_param<sycl::info::platform, std::string,
                           sycl::info::platform::name>(log, plt);
      check_get_info_param<sycl::info::platform, std::string,
                           sycl::info::platform::vendor>(log, plt);
      check_get_info_param<sycl::info::platform, std::vector<std::string>,
                           sycl::info::platform::extensions>(log, plt);
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace platform_info__ */
