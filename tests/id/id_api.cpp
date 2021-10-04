/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME id_api

namespace id_api__ {
using namespace sycl_cts;

template <int dims>
class test_kernel {};

template <int dims>
void test_id_kernels(
    sycl::id<dims> id,
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::target::device>
        error_ptr,
    int m_iteration) {
  sycl::id<dims> id_two(id * 2);
  sycl::id<dims> id_three(id);
  size_t integer = 16;
  for (int j = 0; j < dims; j++) {
    if (id_two.get(j) == 0) {
      id_two[j] = 1;
    }
  }
  const sycl::id<dims> id_two_const(id_two);
  const sycl::id<dims> id_const(id);

  // operators
  // +=
  INDEX_ASSIGNMENT_TESTS(+=, +, id, id_two, id_three);

  // -=
  INDEX_ASSIGNMENT_TESTS(-=, -, id, id_two, id_three);

  // *=
  INDEX_ASSIGNMENT_TESTS(*=, *, id, id_two, id_three);

  // /=
  INDEX_ASSIGNMENT_TESTS(/=, /, id, id_two, id_three);

  // %=
  INDEX_ASSIGNMENT_TESTS(%=, %, id, id_two, id_three);

  // >>=
  INDEX_ASSIGNMENT_TESTS(>>=, >>, id, id_two, id_three);

  // <<=
  INDEX_ASSIGNMENT_TESTS(<<=, <<, id, id_two, id_three);

  // &=
  INDEX_ASSIGNMENT_TESTS(&=, &, id, id_two, id_three);

  // |=
  INDEX_ASSIGNMENT_TESTS(|=, |, id, id_two, id_three);

  // ^=
  INDEX_ASSIGNMENT_TESTS(^=, ^, id, id_two, id_three);

  // check id<dimensions> operatorOP(const id<dimensions> &rhs)

  // *
  INDEX_KERNEL_TEST(*, id, id_two_const, id_three);

  // /
  INDEX_KERNEL_TEST(/, id, id_two_const, id_three);

  //+
  INDEX_KERNEL_TEST(+, id, id_two_const, id_three);

  //-
  INDEX_KERNEL_TEST(-, id, id_two_const, id_three);

  //%
  INDEX_KERNEL_TEST(%, id, id_two_const, id_three);

  //<<
  INDEX_KERNEL_TEST(<<, id, id_two_const, id_three);

  //>>
  INDEX_KERNEL_TEST(>>, id, id_two_const, id_three);

  //&
  INDEX_KERNEL_TEST(&, id, id_two_const, id_three);

  //|
  INDEX_KERNEL_TEST(|, id, id_two_const, id_three);

  //^
  INDEX_KERNEL_TEST (^, id, id_two_const, id_three);

  // &&
  INDEX_KERNEL_TEST(&&, id, id_two_const, id_three);

  // ||
  INDEX_KERNEL_TEST(||, id, id_two_const, id_three);

  // >
  INDEX_KERNEL_TEST(>, id, id_two_const, id_three);

  // <
  INDEX_KERNEL_TEST(<, id, id_two_const, id_three);

  // >=
  INDEX_KERNEL_TEST(>=, id, id_two_const, id_three);

  // <=
  INDEX_KERNEL_TEST(<=, id, id_two_const, id_three);

  // check == and !=
  // ==
  INDEX_EQ_KERNEL_TEST(==, id, id_two);

  // !=
  INDEX_EQ_KERNEL_TEST(!=, id, id_two);

  // check id<dimensions> operatorOP(const size_t &rhs)

  // *
  DUAL_SIZE_INDEX_KERNEL_TEST(*, id, integer, id_three);

  // +
  DUAL_SIZE_INDEX_KERNEL_TEST(+, id, integer, id_three);

  // -
  DUAL_SIZE_INDEX_KERNEL_TEST(-, id, integer, id_three);

  // /
  DUAL_SIZE_INDEX_KERNEL_TEST(/, id, integer, id_three);

  // %
  DUAL_SIZE_INDEX_KERNEL_TEST(%, id, integer, id_three);

  // <<
  DUAL_SIZE_INDEX_KERNEL_TEST(<<, id, integer, id_three);

  // >>
  DUAL_SIZE_INDEX_KERNEL_TEST(>>, id, integer, id_three);

  // |
  DUAL_SIZE_INDEX_KERNEL_TEST(|, id, integer, id_three);

  // ^
  DUAL_SIZE_INDEX_KERNEL_TEST (^, id, integer, id_three);

  // && id can only be lhs
  INDEX_SIZE_T_KERNEL_TEST(&&, id, integer, id_three);

  // || id can only be lhs
  INDEX_SIZE_T_KERNEL_TEST(||, id, integer, id_three);

  // <
  DUAL_SIZE_INDEX_KERNEL_TEST(<, id, integer, id_three);

  // >
  DUAL_SIZE_INDEX_KERNEL_TEST(>, id, integer, id_three);

  // <=
  DUAL_SIZE_INDEX_KERNEL_TEST(<=, id, integer, id_three);

  // >=
  DUAL_SIZE_INDEX_KERNEL_TEST(>=, id, integer, id_three);

  // check id<dimensions> &operatorOP(const size_t &rhs)

  // +=
  INDEX_ASSIGNMENT_INTEGER_TESTS(+=, +, id, integer, id_three);

  // -=
  INDEX_ASSIGNMENT_INTEGER_TESTS(-=, -, id, integer, id_three);

  // *=
  INDEX_ASSIGNMENT_INTEGER_TESTS(*=, *, id, integer, id_three);

  // /=
  INDEX_ASSIGNMENT_INTEGER_TESTS(/=, /, id, integer, id_three);

  // %=
  INDEX_ASSIGNMENT_INTEGER_TESTS(%=, %, id, integer, id_three);

  // >>=
  INDEX_ASSIGNMENT_INTEGER_TESTS(>>=, >>, id, integer, id_three);

  // <<=
  INDEX_ASSIGNMENT_INTEGER_TESTS(<<=, <<, id, integer, id_three);

  // &=
  INDEX_ASSIGNMENT_INTEGER_TESTS(&=, &, id, integer, id_three);

  // |=
  INDEX_ASSIGNMENT_INTEGER_TESTS(|=, |, id, integer, id_three);

  // ^=
  INDEX_ASSIGNMENT_INTEGER_TESTS(^=, ^, id, integer, id_three);
}

template <int dims>
class test_id {
 public:
  // golden values
  static const int m_x = 16;
  static const int m_y = 32;
  static const int m_z = 64;
  static const int m_local = 2;
  static const int error_size = 200;  // up to 200 possible errors
  int m_error[error_size];

  void operator()(util::logger &log, sycl::range<dims> global,
                  sycl::range<dims> local, sycl::queue q) {
    // for testing get()
    for (int i = 0; i < error_size; i++) {
      m_error[i] = 0;  // no error
    }

    {
      sycl::buffer<int, 1> error_buffer(m_error,
                                            sycl::range<1>(error_size));

      q.submit([&](sycl::handler &cgh) {
        auto my_range = sycl::nd_range<dims>(global, local);

        auto error_ptr =
            error_buffer.get_access<sycl::access_mode::read_write>(cgh);

        auto my_kernel = ([=](sycl::nd_item<dims> item) {
          int m_iteration = 0;

          // create check table
          sycl::id<dims> id = item.get_nd_range().get_global_range();

          size_t check[] = {m_x, m_y, m_z};

          for (int i = 0; i < dims; i++) {
            if (id.get(i) > check[i] || id[i] > check[i]) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          }

          test_id_kernels<dims>(id, error_ptr,
                                m_iteration);  // test all in the kernel
        });
        cgh.parallel_for<class test_kernel<dims>>(my_range, my_kernel);
      });

      q.wait_and_throw();
    }
    for (int i = 0; i < error_size; i++) {
      CHECK_VALUE(log, m_error[i], 0, i);
    }
  }
};

/** test sycl::range::get(int index) return size_t
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
      // use across all the dimensions
      auto my_queue = util::get_cts_object::queue();
      // templated approach
      {
        sycl::range<1> range_1d_g(test_id<1>::m_x);
        sycl::range<2> range_2d_g(test_id<2>::m_x, test_id<2>::m_y);
        sycl::range<3> range_3d_g(test_id<3>::m_x, test_id<3>::m_y,
                                      test_id<3>::m_z);

        sycl::range<1> range_1d_l(test_id<1>::m_local);
        sycl::range<2> range_2d_l(test_id<2>::m_local, test_id<2>::m_local);
        sycl::range<3> range_3d_l(test_id<3>::m_local, test_id<3>::m_local,
                                      test_id<3>::m_local);

        test_id<1> test1d;
        test1d(log, range_1d_g, range_1d_l, my_queue);
        test_id<2> test2d;
        test2d(log, range_2d_g, range_2d_l, my_queue);
        test_id<3> test3d;
        test3d(log, range_3d_g, range_3d_l, my_queue);
      }
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace id_api__
