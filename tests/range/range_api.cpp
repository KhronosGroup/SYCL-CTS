/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME range_api

namespace range_api__ {
using namespace sycl_cts;

template <int dims>
class test_range_kernel {};

template <int dims>
void test_range_kernels(
    cl::sycl::range<dims> range,
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer>
        error_ptr,
    int m_iteration) {
  cl::sycl::range<dims> range_two(range * 2);
  cl::sycl::range<dims> range_three(range);
  size_t integer = 16;
  for (int j = 0; j < dims; j++) {
    if (range_two.get(j) == 0) {
      range_two[j] = 1;
    }
  }
  const cl::sycl::range<dims> range_two_const(range_two);
  const cl::sycl::range<dims> range_const(range);

  // operators
  // +=
  INDEX_ASSIGNMENT_TESTS(+=, +, range, range_two, range_three);

  // -=
  INDEX_ASSIGNMENT_TESTS(-=, -, range, range_two, range_three);

  // *=
  INDEX_ASSIGNMENT_TESTS(*=, *, range, range_two, range_three);

  // /=
  INDEX_ASSIGNMENT_TESTS(/=, /, range, range_two, range_three);

  // %=
  INDEX_ASSIGNMENT_TESTS(%=, %, range, range_two, range_three);

  // >>=
  INDEX_ASSIGNMENT_TESTS(>>=, >>, range, range_two, range_three);

  // <<=
  INDEX_ASSIGNMENT_TESTS(<<=, <<, range, range_two, range_three);

  // &=
  INDEX_ASSIGNMENT_TESTS(&=, &, range, range_two, range_three);

  // |=
  INDEX_ASSIGNMENT_TESTS(|=, |, range, range_two, range_three);

  // ^=
  INDEX_ASSIGNMENT_TESTS(^=, ^, range, range_two, range_three);

  // check range<dimensions> operatorOP(const range<dimensions> &rhs)

  // *
  INDEX_KERNEL_TEST(*, range, range_two_const, range_three);

  // /
  INDEX_KERNEL_TEST(/, range, range_two_const, range_three);

  //+
  INDEX_KERNEL_TEST(+, range, range_two_const, range_three);

  //-
  INDEX_KERNEL_TEST(-, range, range_two_const, range_three);

  //%
  INDEX_KERNEL_TEST(%, range, range_two_const, range_three);

  //<<
  INDEX_KERNEL_TEST(<<, range, range_two_const, range_three);

  //>>
  INDEX_KERNEL_TEST(>>, range, range_two_const, range_three);

  //&
  INDEX_KERNEL_TEST(&, range, range_two_const, range_three);

  //|
  INDEX_KERNEL_TEST(|, range, range_two_const, range_three);

  //^
  INDEX_KERNEL_TEST (^, range, range_two_const, range_three);

  // &&
  INDEX_KERNEL_TEST(&&, range, range_two_const, range_three);

  // ||
  INDEX_KERNEL_TEST(||, range, range_two_const, range_three);

  // >
  INDEX_KERNEL_TEST(>, range, range_two_const, range_three);

  // <
  INDEX_KERNEL_TEST(<, range, range_two_const, range_three);

  // >=
  INDEX_KERNEL_TEST(>=, range, range_two_const, range_three);

  // <=
  INDEX_KERNEL_TEST(<=, range, range_two_const, range_three);

  // check == and !=
  // ==
  INDEX_EQ_KERNEL_TEST(==, range, range_two);

  // !=
  INDEX_EQ_KERNEL_TEST(!=, range, range_two);

  // check range<dimensions> operatorOP(const size_t &rhs)

  // *
  DUAL_SIZE_INDEX_KERNEL_TEST(*, range, integer, range_three);

  // +
  DUAL_SIZE_INDEX_KERNEL_TEST(+, range, integer, range_three);

  // -
  DUAL_SIZE_INDEX_KERNEL_TEST(-, range, integer, range_three);

  // /
  DUAL_SIZE_INDEX_KERNEL_TEST(/, range, integer, range_three);

  // %
  DUAL_SIZE_INDEX_KERNEL_TEST(%, range, integer, range_three);

  // <<
  DUAL_SIZE_INDEX_KERNEL_TEST(<<, range, integer, range_three);

  // >>
  DUAL_SIZE_INDEX_KERNEL_TEST(>>, range, integer, range_three);

  // |
  DUAL_SIZE_INDEX_KERNEL_TEST(|, range, integer, range_three);

  // ^
  DUAL_SIZE_INDEX_KERNEL_TEST (^, range, integer, range_three);

  // && range can only be lhs
  INDEX_SIZE_T_KERNEL_TEST(&&, range, integer, range_three);

  // || range can only be lhs
  INDEX_SIZE_T_KERNEL_TEST(||, range, integer, range_three);

  // <
  DUAL_SIZE_INDEX_KERNEL_TEST(<, range, integer, range_three);

  // >
  DUAL_SIZE_INDEX_KERNEL_TEST(>, range, integer, range_three);

  // <=
  DUAL_SIZE_INDEX_KERNEL_TEST(<=, range, integer, range_three);

  // >=
  DUAL_SIZE_INDEX_KERNEL_TEST(>=, range, integer, range_three);

  // check range<dimensions> &operatorOP(const size_t &rhs)

  // +=
  INDEX_ASSIGNMENT_INTEGER_TESTS(+=, +, range, integer, range_three);

  // -=
  INDEX_ASSIGNMENT_INTEGER_TESTS(-=, -, range, integer, range_three);

  // *=
  INDEX_ASSIGNMENT_INTEGER_TESTS(*=, *, range, integer, range_three);

  // /=
  INDEX_ASSIGNMENT_INTEGER_TESTS(/=, /, range, integer, range_three);

  // %=
  INDEX_ASSIGNMENT_INTEGER_TESTS(%=, %, range, integer, range_three);

  // >>=
  INDEX_ASSIGNMENT_INTEGER_TESTS(>>=, >>, range, integer, range_three);

  // <<=
  INDEX_ASSIGNMENT_INTEGER_TESTS(<<=, <<, range, integer, range_three);

  // &=
  INDEX_ASSIGNMENT_INTEGER_TESTS(&=, &, range, integer, range_three);

  // |=
  INDEX_ASSIGNMENT_INTEGER_TESTS(|=, |, range, integer, range_three);

  // ^=
  INDEX_ASSIGNMENT_INTEGER_TESTS(^=, ^, range, integer, range_three);
}

template <int dims>
class test_range {
 public:
  // golden values
  static const int m_x = 16;
  static const int m_y = 32;
  static const int m_z = 64;
  static const int m_local = 2;
  static const int error_size = 204; // up to 204 possible errors
  int m_error[error_size];

  void operator()(util::logger &log, cl::sycl::range<dims> global,
                  cl::sycl::range<dims> local, cl::sycl::queue q) {
    // for testing get()
    for (int i = 0; i < error_size; i++) {
      m_error[i] = 0;  // no error
    }

    {
      cl::sycl::buffer<int, 1> error_buffer(m_error,
                                            cl::sycl::range<1>(error_size));

      q.submit([&](cl::sycl::handler &cgh) {
        auto my_range = cl::sycl::nd_range<dims>(global, local);

        auto error_ptr =
            error_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);

        auto my_kernel = ([=](cl::sycl::nd_item<dims> item) {
          int m_iteration = 0;

          // create check table
          cl::sycl::range<dims> range = item.get_nd_range().get_global_range();

          size_t check[] = {m_x, m_y, m_z};

          if (dims == 1) {
            if (range.size() != m_x) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          } else if (dims == 2) {
            if (range.size() != m_x * m_y) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          } else if (dims == 3) {
            if (range.size() != m_x * m_y * m_z) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          }

          for (int i = 0; i < dims; i++) {
            if (range.get(i) > check[i] || range[i] > check[i]) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          }

          test_range_kernels<dims>(range, error_ptr,
                                   m_iteration);  // test all in the kernel
        });
        cgh.parallel_for<class test_range_kernel<dims>>(my_range, my_kernel);
      });

      q.wait_and_throw();
    }
    for (int i = 0; i < error_size; i++) {
      CHECK_VALUE(log, m_error[i], 0, i);
    }
  }
};

/** test cl::sycl::range::get(int index) return size_t
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
      // use across all the dimensions
      auto my_queue = util::get_cts_object::queue();
      // templated approach
      {
        cl::sycl::range<1> range_1d_g(test_range<1>::m_x);
        cl::sycl::range<2> range_2d_g(test_range<2>::m_x, test_range<2>::m_y);
        cl::sycl::range<3> range_3d_g(test_range<3>::m_x, test_range<3>::m_y,
                                      test_range<3>::m_z);

        cl::sycl::range<1> range_1d_l(test_range<1>::m_local);
        cl::sycl::range<2> range_2d_l(test_range<2>::m_local,
                                      test_range<2>::m_local);
        cl::sycl::range<3> range_3d_l(test_range<3>::m_local,
                                      test_range<3>::m_local,
                                      test_range<3>::m_local);

        test_range<1> test1d;
        test1d(log, range_1d_g, range_1d_l, my_queue);
        test_range<2> test2d;
        test2d(log, range_2d_g, range_2d_l, my_queue);
        test_range<3> test3d;
        test3d(log, range_3d_g, range_3d_l, my_queue);
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace range_api__ */
