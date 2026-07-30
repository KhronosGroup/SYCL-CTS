/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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
    {
      // FIXME: Reenable when struct information descriptors are implemented
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
      /** check get_info parameters
       */
      {
        auto plt = util::get_cts_object::platform(cts_selector);
        check_get_info_param<sycl::info::platform::version, std::string>(log,
                                                                         plt);
        check_get_info_param<sycl::info::platform::name, std::string>(log, plt);
        check_get_info_param<sycl::info::platform::vendor, std::string>(log,
                                                                        plt);
        check_get_info_param<sycl::info::platform::extensions,
                             std::vector<std::string>>(log, plt);
      }
#endif
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace platform_info__ */
