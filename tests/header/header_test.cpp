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

#define TEST_NAME header_test

namespace header_test__ {
using namespace sycl_cts;

/** test SYCL header for compilation and macro definitions
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
  void run(util::logger& log) override {
/** checks that __FAST_RELAXED_MATH__ is defined
*/
#if defined(__FAST_RELAXED_MATH__)
    log.note("__FAST_RELAXED_MATH__ defined");
#endif

/** checks that __SYCL_DEVICE_ONLY__ is defined
*/
#if defined(__SYCL_DEVICE_ONLY__)
    log.note("__SYCL_DEVICE_ONLY__ defined");
#endif

/** checks that __SYCL_SINGLE_SOURCE__ is defined
*/
#if defined(__SYCL_SINGLE_SOURCE__)
    log.note("__SYCL_SINGLE_SOURCE__ defined");
#endif

#if defined(TEST_FAIL)
    FAIL(log, "sycl macro undefined, see previous error");
#endif
  }
};

TEST_CASE(
    "The implementation defines the correct SYCL_LANGUAGE_VERSION macro") {
#ifndef SYCL_LANGUAGE_VERSION
  FAIL("SYCL_LANGUAGE_VERSION is not defined");
#else
  STATIC_REQUIRE(std::is_same_v<decltype(SYCL_LANGUAGE_VERSION), long>);
  STATIC_REQUIRE(SYCL_LANGUAGE_VERSION == 202012L);
#endif
}

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace header_test__ */
