/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "multi_ptr_api_common.h"

#define TEST_NAME multi_ptr_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace multi_ptr_api_common;

/** tests the api for explicit pointers
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
    try {
      auto queue = util::get_cts_object::queue();

      pointer_apis<int, void> voidTests;
      voidTests(log, queue);

      pointer_apis<const int, const void> constVoidTests;
      constVoidTests(log, queue);

      pointer_apis<int> intTests;
      intTests(log, queue);

      pointer_apis<const int> constIntTests;
      constIntTests(log, queue);

      pointer_apis<unsigned int, void> voidUintTests;
      voidUintTests(log, queue);

      pointer_apis<const unsigned int, const void> constUintVoidTests;
      constUintVoidTests(log, queue);

      pointer_apis<unsigned int> uintTests;
      uintTests(log, queue);

      pointer_apis<const unsigned int> constUintTests;
      constUintTests(log, queue);

      pointer_apis<char, void> charVoidTests;
      charVoidTests(log, queue);

      pointer_apis<const char, const void> constCharVoidTests;
      constCharVoidTests(log, queue);

      pointer_apis<char> charTests;
      charTests(log, queue);

      pointer_apis<const char> constCharTests;
      constCharTests(log, queue);

      pointer_apis<short int, void> shortVoidTests;
      shortVoidTests(log, queue);

      pointer_apis<const short int, const void> constShortVoidTests;
      constShortVoidTests(log, queue);

      pointer_apis<short int> shortTests;
      shortTests(log, queue);

      pointer_apis<const short int> constShortTests;
      constShortTests(log, queue);

      pointer_apis<long int, void> longVoidTests;
      longVoidTests(log, queue);

      pointer_apis<const long int, const void> constLongVoidTests;
      constLongVoidTests(log, queue);

      pointer_apis<long int> longTests;
      longTests(log, queue);

      pointer_apis<const long int> constLongTests;
      constLongTests(log, queue);

      pointer_apis<long long int, void> longLongVoidTests;
      longLongVoidTests(log, queue);

      pointer_apis<const long long int, const void> constLongLongVoidTests;
      constLongLongVoidTests(log, queue);

      pointer_apis<long long int> longLongTests;
      longLongTests(log, queue);

      pointer_apis<const long long int> constLongLongTests;
      constLongLongTests(log, queue);

      pointer_apis<float, void> floatVoidTests;
      floatVoidTests(log, queue);

      pointer_apis<const float, const void> constFloatVoidTests;
      constFloatVoidTests(log, queue);

      pointer_apis<const float> constFloatTests;
      constFloatTests(log, queue);

      pointer_apis<float> floatTests;
      floatTests(log, queue);

      pointer_apis<double, void> doubleVoidTests;
      doubleVoidTests(log, queue);

      pointer_apis<const double, const void> constDoubleVoidTests;
      constDoubleVoidTests(log, queue);

      pointer_apis<double> doubleTests;
      doubleTests(log, queue);

      pointer_apis<const double> constDoubleTests;
      constDoubleTests(log, queue);

      pointer_apis<unsigned char, void> ucharVoidTests;
      ucharVoidTests(log, queue);

      pointer_apis<const unsigned char, const void> constUcharVoidTests;
      constUcharVoidTests(log, queue);

      pointer_apis<unsigned char> ucharTests;
      ucharTests(log, queue);

      pointer_apis<const unsigned char> constUcharTests;
      constUcharTests(log, queue);

      pointer_apis<bool, void> boolVoidTests;
      boolVoidTests(log, queue);

      pointer_apis<const bool, const void> constBoolVoidTests;
      constBoolVoidTests(log, queue);

      pointer_apis<bool> boolTests;
      boolTests(log, queue);

      pointer_apis<const bool> constBoolTests;
      constBoolTests(log, queue);

      pointer_apis<user_struct> userStructTests;
      userStructTests(log, queue);

      pointer_apis<const user_struct> constUserStructTests;
      constUserStructTests(log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
