/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "multi_ptr_constructors_common.h"

#define TEST_NAME multi_ptr_constructors

namespace TEST_NAMESPACE {
using namespace multi_ptr_constructors_common;
using namespace sycl_cts;

/** tests the constructors for explicit pointers
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

      pointer_ctors<int, void> voidTests;
      voidTests(queue);

      pointer_ctors<const int, const void> constVoidTests;
      constVoidTests(queue);

      pointer_ctors<int> intTests;
      intTests(queue);

      pointer_ctors<const int> constIntTests;
      constIntTests(queue);

      pointer_ctors<unsigned int, void> voidUintTests;
      voidUintTests(queue);

      pointer_ctors<const unsigned int, const void> constUintVoidTests;
      constUintVoidTests(queue);

      pointer_ctors<unsigned int> uintTests;
      uintTests(queue);

      pointer_ctors<const unsigned int> constUintTests;
      constUintTests(queue);

      pointer_ctors<char, void> charVoidTests;
      charVoidTests(queue);

      pointer_ctors<const char, const void> constCharVoidTests;
      constCharVoidTests(queue);

      pointer_ctors<char> charTests;
      charTests(queue);

      pointer_ctors<const char> constCharTests;
      constCharTests(queue);

      pointer_ctors<short int, void> shortVoidTests;
      shortVoidTests(queue);

      pointer_ctors<const short int, const void> constShortVoidTests;
      constShortVoidTests(queue);

      pointer_ctors<short int> shortTests;
      shortTests(queue);

      pointer_ctors<const short int> constShortTests;
      constShortTests(queue);

      pointer_ctors<long int, void> longVoidTests;
      longVoidTests(queue);

      pointer_ctors<const long int, const void> constLongVoidTests;
      constLongVoidTests(queue);

      pointer_ctors<long int> longTests;
      longTests(queue);

      pointer_ctors<const long int> constLongTests;
      constLongTests(queue);

      pointer_ctors<long long int, void> longLongVoidTests;
      longLongVoidTests(queue);

      pointer_ctors<const long long int, const void> constLongLongVoidTests;
      constLongLongVoidTests(queue);

      pointer_ctors<long long int> longLongTests;
      longLongTests(queue);

      pointer_ctors<const long long int> constLongLongTests;
      constLongLongTests(queue);

      pointer_ctors<float, void> floatVoidTests;
      floatVoidTests(queue);

      pointer_ctors<const float, const void> constFloatVoidTests;
      constFloatVoidTests(queue);

      pointer_ctors<const float> constFloatTests;
      constFloatTests(queue);

      pointer_ctors<float> floatTests;
      floatTests(queue);

      pointer_ctors<double, void> doubleVoidTests;
      doubleVoidTests(queue);

      pointer_ctors<const double, const void> constDoubleVoidTests;
      constDoubleVoidTests(queue);

      pointer_ctors<double> doubleTests;
      doubleTests(queue);

      pointer_ctors<const double> constDoubleTests;
      constDoubleTests(queue);

      pointer_ctors<unsigned char, void> ucharVoidTests;
      ucharVoidTests(queue);

      pointer_ctors<const unsigned char, const void> constUcharVoidTests;
      constUcharVoidTests(queue);

      pointer_ctors<unsigned char> ucharTests;
      ucharTests(queue);

      pointer_ctors<const unsigned char> constUcharTests;
      constUcharTests(queue);

      pointer_ctors<bool, void> boolVoidTests;
      boolVoidTests(queue);

      pointer_ctors<const bool, const void> constBoolVoidTests;
      constBoolVoidTests(queue);

      pointer_ctors<bool> boolTests;
      boolTests(queue);

      pointer_ctors<const bool> constBoolTests;
      constBoolTests(queue);

      pointer_ctors<user_struct> userStructTests;
      userStructTests(queue);

      pointer_ctors<const user_struct> constUserStructTests;
      constUserStructTests(queue);

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
