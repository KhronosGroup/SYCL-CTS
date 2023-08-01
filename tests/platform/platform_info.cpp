/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
      /** check get_info parameters
       */
      {
        auto plt = util::get_cts_object::platform(cts_selector);
        check_get_info_param<sycl::info::platform::profile, std::string>(log,
                                                                         plt);
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
