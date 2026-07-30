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

#define TEST_NAME queue_info

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test the info for sycl::queue
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
        auto queue = util::get_cts_object::queue(cts_selector);
        check_get_info_param<sycl::info::queue::context, sycl::context>(log,
                                                                        queue);
        check_get_info_param<sycl::info::queue::device, sycl::device>(log,
                                                                      queue);
      }
#endif
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
