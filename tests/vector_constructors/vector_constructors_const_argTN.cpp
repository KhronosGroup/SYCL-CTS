/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../common/common_vec.h"

#define TEST_NAME vector_constructors_const_argTN

namespace vector_constructors_const_argTN__ {
using namespace sycl_cts;

class VEC_CONST_ARGTN_CONSTRUCTOR_KERNEL;

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
          cl::sycl::buffer<bool, 1> boolBuffer(resArray, cl::sycl::range<1>(1));
          testQueue.submit([&](cl::sycl::handler &cgh) {
            auto resAcc =
                boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

            cgh.single_task<class VEC_CONST_ARGTN_CONSTRUCTOR_KERNEL>([=]() {
              /** Test
               *  template <typename... argTN>
               *  vec(const argTN&... args)
               */
              {
                auto test = cl::sycl::vec<char, 1>(1);
                char vals[] = {1};
                if (!check_equal_type_bool<cl::sycl::vec<char, 1>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<char, 1>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<char, 1>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<unsigned char, 2>(1, 2);
                unsigned char vals[] = {1, 2};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned char, 2>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned char, 2>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned char, 2>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<short, 3>(1, 2, 3);
                short vals[] = {1, 2, 3};
                if (!check_equal_type_bool<cl::sycl::vec<short, 3>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<short, 3>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<short, 3>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<unsigned short, 4>(1, 2, 3, 4);
                unsigned short vals[] = {1, 2, 3, 4};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned short, 4>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned short, 4>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned short, 4>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<int, 8>(1, 2, 3, 4, 5, 6, 7, 8);
                int vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
                if (!check_equal_type_bool<cl::sycl::vec<int, 8>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<int, 8>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<int, 8>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<unsigned int, 16>(
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
                unsigned int vals[] = {1, 2,  3,  4,  5,  6,  7,  8,
                                       9, 10, 11, 12, 13, 14, 15, 16};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned int, 16>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned int, 16>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned int, 16>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<long, 1>(1);
                long vals[] = {1};
                if (!check_equal_type_bool<cl::sycl::vec<long, 1>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<long, 1>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<long, 1>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<unsigned long, 2>(1, 2);
                unsigned long vals[] = {1, 2};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned long, 2>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned long, 2>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned long, 2>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<long long, 3>(1, 2, 3);
                long long vals[] = {1, 2, 3};
                if (!check_equal_type_bool<cl::sycl::vec<long long, 3>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<long long, 3>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<long long, 3>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<unsigned long long, 4>(1, 2, 3, 4);
                unsigned long long vals[] = {1, 2, 3, 4};
                if (!check_equal_type_bool<
                        cl::sycl::vec<unsigned long long, 4>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned long long, 4>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned long long, 4>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<float, 8>(1.0f, 2.0f, 3.0f, 4.0f,
                                                    5.0f, 6.0f, 7.0f, 8.0f);
                float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                if (!check_equal_type_bool<cl::sycl::vec<float, 8>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<float, 8>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<float, 8>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto test = cl::sycl::vec<double, 16>(
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                    12.0, 13.0, 14.0, 15.0, 16.0);
                double vals[] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
                if (!check_equal_type_bool<cl::sycl::vec<double, 16>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<double, 16>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<double, 16>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec = cl::sycl::vec<unsigned char, 1>(1);
                auto test = cl::sycl::vec<unsigned char, 2>(anotherVec, 2);
                unsigned char vals[] = {1, 2};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned char, 2>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned char, 2>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned char, 2>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec = cl::sycl::vec<short, 1>(2);
                auto test = cl::sycl::vec<short, 3>(1, anotherVec, 3);
                short vals[] = {1, 2, 3};
                if (!check_equal_type_bool<cl::sycl::vec<short, 3>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<short, 3>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<short, 3>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec = cl::sycl::vec<unsigned short, 1>(4);
                auto test =
                    cl::sycl::vec<unsigned short, 4>(1, 2, 3, anotherVec);
                unsigned short vals[] = {1, 2, 3, 4};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned short, 4>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned short, 4>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned short, 4>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec1 = cl::sycl::vec<int, 2>(1, 2);
                auto anotherVec2 = cl::sycl::vec<int, 1>(4);
                auto anotherVec3 = cl::sycl::vec<int, 3>(6, 7, 8);
                auto test = cl::sycl::vec<int, 8>(anotherVec1, 3, anotherVec2,
                                                  5, anotherVec3);
                int vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
                if (!check_equal_type_bool<cl::sycl::vec<int, 8>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<int, 8>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<int, 8>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec1 =
                    cl::sycl::vec<unsigned int, 8>(1, 2, 3, 4, 5, 6, 7, 8);
                auto anotherVec2 = cl::sycl::vec<unsigned int, 8>(
                    9, 10, 11, 12, 13, 14, 15, 16);
                auto test =
                    cl::sycl::vec<unsigned int, 16>(anotherVec1, anotherVec2);
                unsigned int vals[] = {1, 2,  3,  4,  5,  6,  7,  8,
                                       9, 10, 11, 12, 13, 14, 15, 16};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned int, 16>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned int, 16>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned int, 16>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec = cl::sycl::vec<unsigned long, 1>(2);
                auto test = cl::sycl::vec<unsigned long, 2>(1, anotherVec);
                unsigned long vals[] = {1, 2};
                if (!check_equal_type_bool<cl::sycl::vec<unsigned long, 2>>(
                        test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned long, 2>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned long, 2>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec = cl::sycl::vec<long long, 1>(3);
                auto test = cl::sycl::vec<long long, 3>(1, 2, anotherVec);
                long long vals[] = {1, 2, 3};
                if (!check_equal_type_bool<cl::sycl::vec<long long, 3>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<long long, 3>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<long long, 3>(test, vals)) {
                  resAcc[0] = false;
                }
              }
              {
                auto anotherVec1 = cl::sycl::vec<unsigned long long, 1>(1);
                auto anotherVec2 = cl::sycl::vec<unsigned long long, 1>(2);
                auto anotherVec3 = cl::sycl::vec<unsigned long long, 1>(3);
                auto anotherVec4 = cl::sycl::vec<unsigned long long, 1>(4);
                auto test = cl::sycl::vec<unsigned long long, 4>(
                    anotherVec1, anotherVec2, anotherVec3, anotherVec4);
                unsigned long long vals[] = {1, 2, 3, 4};
                if (!check_equal_type_bool<
                        cl::sycl::vec<unsigned long long, 4>>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_size<unsigned long long, 4>(test)) {
                  resAcc[0] = false;
                }
                if (!check_vector_values<unsigned long long, 4>(test, vals)) {
                  resAcc[0] = false;
                }
              }
            });
          });
          if (!resArray[0]) {
            fail_test(log,
                      cl::sycl::string_class(
                          "A vec(const &vecT) constructor test has failed"));
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

} /* namespace vector_constructors_const_argTN__ */
