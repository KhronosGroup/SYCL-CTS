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

/** Performs a test of each vector operator available to all types,
   *  on a given type, size, and two given values of that type
   */
template <typename vecType, int vecSize>
bool test_all_type_vector_operators(vecType testValue1, vecType testValue2) {
  auto testVec1 = cl::sycl::vec<vecType, vecSize>(testValue1);
  auto testVec2 = cl::sycl::vec<vecType, vecSize>(testValue2);
  cl::sycl::vec<vecType, vecSize> resVec;

  // Arithmetic operators
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 + testValue2);
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 + testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 + testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 - testValue2);
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 - testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 - testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 * testValue2);
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 * testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 * testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 / testValue2);
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 / testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 / testValue2; })) {
    return false;
  }

  // Post and pre increment and decrement
  auto tempTest = testValue1;
  resVec = cl::sycl::vec<vecType, vecSize>(++tempTest);
  if (!check_single_vector_op<vecSize>(resVec, [=]() mutable {
        testVec1++;
        return testVec1;
      })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(resVec, [=]() mutable {
        ++testVec1;
        return testVec1;
      })) {
    return false;
  }
  tempTest = testValue1;
  resVec = cl::sycl::vec<vecType, vecSize>(--tempTest);
  if (!check_single_vector_op<vecSize>(resVec, [=]() mutable {
        testVec1--;
        return testVec1;
      })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(resVec, [=]() mutable {
        --testVec1;
        return testVec1;
      })) {
    return false;
  }

  // Assignment operators
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 + testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 += testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 += testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 - testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 -= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 -= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 * testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 *= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 *= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 / testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 /= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 /= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 = testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 = testValue2; })) {
    return false;
  }
  return true;
}

/** Performs a test of each vector operator not available to floating point
 *  types, on a given type, size, and two given values of that type
 */
template <typename vecType, int vecSize>
bool test_non_fp_vector_operators(vecType testValue1, vecType testValue2) {
  auto testVec1 = cl::sycl::vec<vecType, vecSize>(testValue1);
  auto testVec2 = cl::sycl::vec<vecType, vecSize>(testValue2);
  cl::sycl::vec<vecType, vecSize> resVec;

  // Arithmetic operations
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 % testValue2);
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 % testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 % testValue2; })) {
    return false;
  }

  // Bitwise operations
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 >> testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 >> testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 >> testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 << testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 << testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 << testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(~testValue1);
  if (!check_single_vector_op<vecSize>(resVec, [=]() { return ~testVec1; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 | testValue2));
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 | testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 | testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 ^ testValue2));
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 ^ testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 ^ testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 & testValue2));
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 & testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 & testValue2; })) {
    return false;
  }

  // Assignment operations
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 % testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 %= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 %= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 | testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 |= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 |= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 ^ testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 ^= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 ^= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 << testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 <<= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 <<= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 >> testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 >>= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 >>= testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(testValue1 & testValue2);
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 &= testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() mutable { return testVec1 &= testValue2; })) {
    return false;
  }
  return true;
}

/** Tests each logical and relational operator available to vector types
 */
template <typename retType, typename vecType, int vecSize>
bool test_specific_return_type_vector_operators(vecType testValue1,
                                                vecType testValue2) {
  auto testVec1 = cl::sycl::vec<vecType, vecSize>(testValue1);
  auto testVec2 = cl::sycl::vec<vecType, vecSize>(testValue2);
  cl::sycl::vec<retType, vecSize> resVec;

  // Logical operators
  resVec = cl::sycl::vec<int, vecSize>(-(testValue1 && testValue2));
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 && testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 && testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<int, vecSize>(-(testValue1 || testValue2));
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 || testVec2; })) {
    return false;
  }
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 || testValue2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(!testValue1);
  if (!check_single_vector_op<vecSize>(resVec, [=]() { return !testVec1; })) {
    return false;
  }

  // Relational Operators
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 == testValue2));
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 == testVec2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 != testValue2));
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 != testVec2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 <= testValue2));
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 <= testVec2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 >= testValue2));
  if (!check_single_vector_op<vecSize>(
          resVec, [=]() { return testVec1 >= testVec2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 < testValue2));
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 < testVec2; })) {
    return false;
  }
  resVec = cl::sycl::vec<vecType, vecSize>(-(testValue1 > testValue2));
  if (!check_single_vector_op<vecSize>(resVec,
                                       [=]() { return testVec1 > testVec2; })) {
    return false;
  }
  return true;
}

/** Test a cross section of vector constructors
 *  This doesn't test every possible combination and type of size of
 *  vector constructor.
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
    try {
      auto testQueue = util::get_cts_object::queue();
      {
        auto testDevice = testQueue.get_device();

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_signedchar>([=]() {

                if (!test_all_type_vector_operators<signed char, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<signed char, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'signed char' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_unsignedchar>([=]() {

                if (!test_all_type_vector_operators<unsigned char, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<unsigned char, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'unsigned char' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_char>([=]() {

                if (!test_all_type_vector_operators<char, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<char, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'char' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_short>([=]() {

                if (!test_all_type_vector_operators<short, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<short, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'short' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_unsignedshort>([=]() {

                if (!test_all_type_vector_operators<unsigned short, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<unsigned short, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'unsigned short' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_int>([=]() {

                if (!test_all_type_vector_operators<int, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<int, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'int' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_unsignedint>([=]() {

                if (!test_all_type_vector_operators<unsigned int, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<unsigned int, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'unsigned int' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));

            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_long>([=]() {

                if (!test_all_type_vector_operators<long, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<long, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'long' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_unsignedlong>([=]() {

                if (!test_all_type_vector_operators<unsigned long, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<unsigned long, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'unsigned long' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_longlong>([=]() {

                if (!test_all_type_vector_operators<long long, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<long long, 2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'long long' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_unsignedlonglong>([=]() {

                if (!test_all_type_vector_operators<unsigned long long, 2>(1,
                                                                           2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<unsigned long long, 2>(1,
                                                                         2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log, cl::sycl::string_class(
                               "A vector operator test for type "
                               "'unsigned long long' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_float>([=]() {

                if (!test_all_type_vector_operators<float, 2>(1.0f, 2.0f)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'float' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_double>([=]() {

                if (!test_all_type_vector_operators<double, 2>(1.0, 2.0)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'double' has failed."));
          }
        }

        {
          if (testDevice.has_extension("cl_khr_fp16")) {
            bool resArray[1] = {true};
            {
              cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                   cl::sycl::range<1>(1));
              testQueue.submit([&](cl::sycl::handler &cgh) {
                auto resAcc =
                    boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

                cgh.single_task<class KERNEL_half>([=]() {

                  if (!test_all_type_vector_operators<cl::sycl::half, 2>(
                          1.0f, 2.0f)) {
                    resAcc[0] = false;
                  }

                });
              });
            }
            if (!resArray[0]) {
              fail_test(
                  log, cl::sycl::string_class("A vector operator test for type "
                                              "'cl::sycl::half' has failed."));
            }
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_char>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_char, 2>(1,
                                                                          2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_char, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_char,
                                                                cl_char, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(
                log, cl::sycl::string_class("A vector operator test for type "
                                            "'cl::sycl::cl_char' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_uchar>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_uchar, 2>(1,
                                                                           2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_uchar, 2>(1,
                                                                         2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_char,
                                                                cl_uchar, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log, cl::sycl::string_class(
                               "A vector operator test for type "
                               "'cl::sycl::cl_uchar' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_short>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_short, 2>(1,
                                                                           2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_short, 2>(1,
                                                                         2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_short,
                                                                cl_short, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log, cl::sycl::string_class(
                               "A vector operator test for type "
                               "'cl::sycl::cl_short' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_ushort>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_ushort, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_ushort, 2>(1,
                                                                          2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_short,
                                                                cl_ushort, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log, cl::sycl::string_class(
                               "A vector operator test for type "
                               "'cl::sycl::cl_ushort' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_int>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_int, 2>(1,
                                                                         2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_int, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_int, cl_int,
                                                                2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class("A vector operator test for type "
                                             "'cl::sycl::cl_int' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_uint>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_uint, 2>(1,
                                                                          2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_uint, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_int, cl_uint,
                                                                2>(1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(
                log, cl::sycl::string_class("A vector operator test for type "
                                            "'cl::sycl::cl_uint' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_long>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_long, 2>(1,
                                                                          2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_long, 2>(1, 2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_long,
                                                                cl_long, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(
                log, cl::sycl::string_class("A vector operator test for type "
                                            "'cl::sycl::cl_long' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_ulong>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_ulong, 2>(1,
                                                                           2)) {
                  resAcc[0] = false;
                }
                if (!test_non_fp_vector_operators<cl::sycl::cl_ulong, 2>(1,
                                                                         2)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_long,
                                                                cl_ulong, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log, cl::sycl::string_class(
                               "A vector operator test for type "
                               "'cl::sycl::cl_ulong' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_float>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_float, 2>(
                        1.0f, 2.0f)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_int,
                                                                cl_float, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log, cl::sycl::string_class(
                               "A vector operator test for type "
                               "'cl::sycl::cl_float' has failed."));
          }
        }

        {
          bool resArray[1] = {true};
          {
            cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                 cl::sycl::range<1>(1));
            testQueue.submit([&](cl::sycl::handler &cgh) {
              auto resAcc =
                  boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

              cgh.single_task<class KERNEL_cl_double>([=]() {

                if (!test_all_type_vector_operators<cl::sycl::cl_double, 2>(
                        1.0, 2.0)) {
                  resAcc[0] = false;
                }
                if (!test_specific_return_type_vector_operators<cl_long,
                                                                cl_double, 2>(
                        1, 2)) {
                  resAcc[0] = false;
                }

              });
            });
          }
          if (!resArray[0]) {
            fail_test(log, cl::sycl::string_class(
                               "A vector operator test for type "
                               "'cl::sycl::cl_double' has failed."));
          }
        }

        {
          if (testDevice.has_extension("cl_khr_fp16")) {
            bool resArray[1] = {true};
            {
              cl::sycl::buffer<bool, 1> boolBuffer(resArray,
                                                   cl::sycl::range<1>(1));
              testQueue.submit([&](cl::sycl::handler &cgh) {
                auto resAcc =
                    boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

                cgh.single_task<class KERNEL_cl_half>([=]() {

                  if (!test_all_type_vector_operators<cl::sycl::cl_half, 2>(
                          1.0f, 2.0f)) {
                    resAcc[0] = false;
                  }
                  if (!test_specific_return_type_vector_operators<cl_short,
                                                                  cl_half, 2>(
                          1, 2)) {
                    resAcc[0] = false;
                  }

                });
              });
            }
            if (!resArray[0]) {
              fail_test(log, cl::sycl::string_class(
                                 "A vector operator test for type "
                                 "'cl::sycl::cl_half' has failed."));
            }
          }
        }
      }
      testQueue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace vector_operators */
