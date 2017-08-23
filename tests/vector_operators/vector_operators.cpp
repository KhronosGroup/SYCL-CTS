/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../common/common_vec.h"

#define TEST_NAME vector_operators

namespace vector_operators__ {
using namespace sycl_cts;

/** Test a cross section of vector constructors
 *  This doesn't test every possible combination and type of size of
 *  vector constructor.
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** Performs a test of each vector operator available to all types,
   *  on a given type, size, and two given values of that type
   */
  template <typename vecType, int vecSize>
  void test_all_type_vector_operators(vecType testValue1, vecType testValue2,
                                      util::logger &log) {
    auto testVec1 = cl::sycl::vec<vecType, vecSize>(testValue1);
    auto testVec2 = cl::sycl::vec<vecType, vecSize>(testValue2);
    cl::sycl::vec<vecType, vecSize> resVec;

    // Arithmetic operators
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 + testValue2);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 + testVec2; },
                                    cl::sycl::string_class("+ vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 + testValue2; },
                                    cl::sycl::string_class("+ scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 - testValue2);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 - testVec2; },
                                    cl::sycl::string_class("- vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 - testValue2; },
                                    cl::sycl::string_class("- scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 * testValue2);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 * testVec2; },
                                    cl::sycl::string_class("* vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 * testValue2; },
                                    cl::sycl::string_class("* scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 / testValue2);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 / testVec2; },
                                    cl::sycl::string_class("/ vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 / testValue2; },
                                    cl::sycl::string_class("/ scalar"));

    // Post and pre increment and decrement
    auto tempTest = testValue1;
    resVec = cl::sycl::vec<vecType, vecSize>(++tempTest);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() mutable {
                                      testVec1++;
                                      return testVec1;
                                    },
                                    cl::sycl::string_class("post ++"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() mutable {
                                      ++testVec1;
                                      return testVec1;
                                    },
                                    cl::sycl::string_class("pre ++"));
    tempTest = testValue1;
    resVec = cl::sycl::vec<vecType, vecSize>(--tempTest);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() mutable {
                                      testVec1--;
                                      return testVec1;
                                    },
                                    cl::sycl::string_class("post --"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() mutable {
                                      --testVec1;
                                      return testVec1;
                                    },
                                    cl::sycl::string_class("pre --"));

    // Logical operators
    resVec = cl::sycl::vec<int, vecSize>(-(testValue1 && testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 && testVec2; },
                                    cl::sycl::string_class("&& vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 && testValue2; },
                                    cl::sycl::string_class("&& scalar"));
    resVec = cl::sycl::vec<int, vecSize>(-(testValue1 || testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 || testVec2; },
                                    cl::sycl::string_class("|| vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 || testValue2; },
                                    cl::sycl::string_class("|| scalar"));

    // Assignment operators
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 + testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 += testVec2; },
        cl::sycl::string_class("+= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 += testValue2; },
        cl::sycl::string_class("+= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 - testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 -= testVec2; },
        cl::sycl::string_class("-= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 -= testValue2; },
        cl::sycl::string_class("-= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 * testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 *= testVec2; },
        cl::sycl::string_class("*= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 *= testValue2; },
        cl::sycl::string_class("*= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 / testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 /= testVec2; },
        cl::sycl::string_class("/= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 /= testValue2; },
        cl::sycl::string_class("/= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 = testVec2; },
        cl::sycl::string_class("= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 = testValue2; },
        cl::sycl::string_class("= scalar"));

    // Relational Operators
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 == testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 == testVec2; },
                                    cl::sycl::string_class("=="));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 != testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 != testVec2; },
                                    cl::sycl::string_class("!="));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 <= testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 <= testVec2; },
                                    cl::sycl::string_class("<="));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 >= testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 >= testVec2; },
                                    cl::sycl::string_class(">="));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 < testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 < testVec2; },
                                    cl::sycl::string_class("<"));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 > testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 > testVec2; },
                                    cl::sycl::string_class(">"));
  }

  /** Performs a test of each vector operator not available to floating point
   *  types, on a given type, size, and two given values of that type
   */
  template <typename vecType, int vecSize>
  void test_non_fp_vector_operators(vecType testValue1, vecType testValue2,
                                    util::logger &log) {
    auto testVec1 = cl::sycl::vec<vecType, vecSize>(testValue1);
    auto testVec2 = cl::sycl::vec<vecType, vecSize>(testValue2);
    cl::sycl::vec<vecType, vecSize> resVec;

    // Arithmetic operations
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 % testValue2);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 % testVec2; },
                                    cl::sycl::string_class("\% vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 % testValue2; },
                                    cl::sycl::string_class("\% scalar"));

    // Bitwise operations
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 >> testValue2);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 >> testVec2; },
                                    cl::sycl::string_class(">> vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 >> testValue2; },
                                    cl::sycl::string_class(">> scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 << testValue2);
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 << testVec2; },
                                    cl::sycl::string_class("<< vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 << testValue2; },
                                    cl::sycl::string_class("<< scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(~testValue1);
    check_single_vector_op<vecSize>(log, resVec, [=]() { return ~testVec1; },
                                    cl::sycl::string_class("~"));
    resVec = cl::sycl::vec<vecType, vecSize>(!testValue1);
    check_single_vector_op<vecSize>(log, resVec, [=]() { return !testVec1; },
                                    cl::sycl::string_class("!"));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 | testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 | testVec2; },
                                    cl::sycl::string_class("| vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 | testValue2; },
                                    cl::sycl::string_class("| scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 ^ testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 ^ testVec2; },
                                    cl::sycl::string_class("^ vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 ^ testValue2; },
                                    cl::sycl::string_class("^ scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 & testValue2));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 & testVec2; },
                                    cl::sycl::string_class("& vec"));
    check_single_vector_op<vecSize>(log, resVec,
                                    [=]() { return testVec1 & testValue2; },
                                    cl::sycl::string_class("& scalar"));

    // Assignment operations
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 % testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 %= testVec2; },
        cl::sycl::string_class("\%= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 %= testValue2; },
        cl::sycl::string_class("\%= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 | testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 |= testVec2; },
        cl::sycl::string_class("|= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 |= testValue2; },
        cl::sycl::string_class("|= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 ^ testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 ^= testVec2; },
        cl::sycl::string_class("^= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 ^= testValue2; },
        cl::sycl::string_class("^= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 << testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 <<= testVec2; },
        cl::sycl::string_class("<<= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 <<= testValue2; },
        cl::sycl::string_class("<<= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 >> testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 >>= testVec2; },
        cl::sycl::string_class(">>= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 >>= testValue2; },
        cl::sycl::string_class(">>= scalar"));
    resVec = cl::sycl::vec<vecType, vecSize>(testValue1 & testValue2);
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 &= testVec2; },
        cl::sycl::string_class("&= vec"));
    check_single_vector_op<vecSize>(
        log, resVec, [=]() mutable { return testVec1 &= testValue2; },
        cl::sycl::string_class("&= scalar"));
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    using namespace cl::sycl;

    try {
      test_all_type_vector_operators<signed char, 2>(1, 2, log);
      test_non_fp_vector_operators<signed char, 2>(1, 2, log);
      test_all_type_vector_operators<unsigned char, 2>(1, 2, log);
      test_non_fp_vector_operators<unsigned char, 2>(1, 2, log);
      test_all_type_vector_operators<short, 2>(1, 2, log);
      test_non_fp_vector_operators<short, 2>(1, 2, log);
      test_all_type_vector_operators<unsigned short, 2>(1, 2, log);
      test_non_fp_vector_operators<unsigned short, 2>(1, 2, log);
      test_all_type_vector_operators<int, 2>(1, 2, log);
      test_non_fp_vector_operators<int, 2>(1, 2, log);
      test_all_type_vector_operators<unsigned int, 2>(1, 2, log);
      test_non_fp_vector_operators<unsigned int, 2>(1, 2, log);
      test_all_type_vector_operators<long, 2>(1, 2, log);
      test_non_fp_vector_operators<long, 2>(1, 2, log);
      test_all_type_vector_operators<unsigned long, 2>(1, 2, log);
      test_non_fp_vector_operators<unsigned long, 2>(1, 2, log);
      test_all_type_vector_operators<long long, 2>(1, 2, log);
      test_non_fp_vector_operators<long long, 2>(1, 2, log);
      test_all_type_vector_operators<unsigned long long, 2>(1, 2, log);
      test_non_fp_vector_operators<unsigned long long, 2>(1, 2, log);
      test_all_type_vector_operators<float, 2>(1, 2, log);
      test_all_type_vector_operators<double, 2>(1, 2, log);
      test_all_type_vector_operators<half, 2>(1, 2, log);
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace vector_operators */
