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
    {
      auto context = util::get_cts_object::context();

      /** check get_devices() method
       */
      {
        auto deviceList = context.get_devices();
        check_return_type<std::vector<sycl::device>>(
            log, deviceList, "get_devices()");
      }

      /** check get_platform() method
       */
      {
        auto platform = context.get_platform();
        check_return_type<sycl::platform>(log, platform, "get_platform()");
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace context_api */
