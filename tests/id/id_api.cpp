/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME id_api

/** \brief tests the result of using operator op with operands lhs lhs and lhs
 * rhs, while storing the results in res.
 */
#define ID_KERNEL_TEST(op, lhs, rhs, res)               \
  {                                                     \
    res = lhs op rhs;                                   \
    for (int k = 0; k < dims; k++) {                    \
      if ((res.get(k) != (lhs.get(k) op rhs.get(k))) || \
          (res[k] != (lhs[k] op rhs[k]))) {             \
        error_ptr[m_iteration] = __LINE__;              \
        m_iteration++;                                  \
      }                                                 \
    }                                                   \
  }

/** \brief tests the result of equality/inequality operator op between id
 * operands lhs and rhs
 */
#define ID_EQ_KERNEL_TEST(op, lhs, rhs)             \
  {                                                 \
    if ((lhs op lhs) != (rhs op rhs)) {             \
      error_ptr[m_iteration] = __LINE__;            \
      m_iteration++;                                \
    }                                               \
    bool result = lhs op rhs;                       \
    for (int k = 0; k < dims; k++) {                \
      if ((result != (lhs.get(k) op rhs.get(k))) || \
          (result != (lhs[k] op rhs[k]))) {         \
        error_ptr[m_iteration] = __LINE__;          \
        m_iteration++;                              \
      }                                             \
    }                                               \
  }

/** \brief tests the result of comparisson operator op between id operands lhs
 * and rhs
 */
#define ID_CMP_KERNEL_TEST(op, lhs, rhs)                                       \
  {                                                                            \
    if ((lhs op lhs) == (rhs op rhs)) {                                        \
      error_ptr[m_iteration] = __LINE__;                                       \
      m_iteration++;                                                           \
    }                                                                          \
    bool result = lhs op rhs;                                                  \
    bool check_fail = false;                                                   \
    for (int k = 0; k < dims; k++) {                                           \
      if (((lhs.get(k) != rhs.get(k)) || (lhs[k] != rhs[k])) && !check_fail) { \
        check_fail = true;                                                     \
      }                                                                        \
      if (((lhs.get(k) op rhs.get(k)) || (lhs[k] op rhs[k])) && check_fail) {  \
        error_ptr[m_iteration] = __LINE__;                                     \
        m_iteration++;                                                         \
      }                                                                        \
    }                                                                          \
  }

/** \brief tests the result of operator op between scalar operand lhs and id
 * operand rhs
 */
#define ID_SIZE_T_KERNEL_TEST(op, id, integer, result) \
  {                                                    \
    result = id op integer;                            \
    for (int k = 0; k < dims; k++) {                   \
      if ((result.get(k) != (id.get(k) op integer)) || \
          (result[k] != (id[k] op integer))) {         \
        error_ptr[m_iteration] = __LINE__;             \
        m_iteration++;                                 \
      }                                                \
    }                                                  \
  }

/** \brief tests the result of operator op between scalar operand lhs and id
 * operand rhs
*/
#define SIZE_T_ID_KERNEL_TEST(op, integer, id, result) \
  {                                                    \
    result = integer op id;                            \
    for (int k = 0; k < dims; k++) {                   \
      if ((result.get(k) != (integer op id.get(k))) || \
          (result[k] != (integer op id[k]))) {         \
        error_ptr[m_iteration] = __LINE__;             \
        m_iteration++;                                 \
      }                                                \
    }                                                  \
  }

/** \brief tests the result of operator op between scalar operand and an id
 * operand in any possible configuration
*/
#define DUAL_SIZE_ID_KERNEL_TEST(op, id, integer, result) \
  ID_SIZE_T_KERNEL_TEST(op, id, integer, result);         \
  SIZE_T_ID_KERNEL_TEST(op, integer, id, result)

/** \brief tests the result of assignment operator op between assigning a to c
 * then use the assignment operator assignment_op with lhs operand c and lhs
 * operand b. Then tests the result using operator op with operands a and b.
*/
#define id_assignment_tests(assignment_op, op, a, b, c)                       \
  {                                                                           \
    c = a;                                                                    \
    c assignment_op b;                                                        \
    for (int k = 0; k < dims; k++) {                                          \
      if ((c.get(k) != (a.get(k) op b.get(k))) || (c[k] != (a[k] op b[k]))) { \
        error_ptr[m_iteration] = __LINE__;                                    \
        m_iteration++;                                                        \
      }                                                                       \
    }                                                                         \
  }

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int dims>
class test_kernel {};

template <int dims>
void test_id_kernels(
    cl::sycl::id<dims> id,
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer>
        error_ptr,
    int m_iteration) {
  cl::sycl::id<dims> id_two(id * 2);
  cl::sycl::id<dims> id_three(id);
  size_t integer = 16;
  for (int j = 0; j < dims; j++) {
    if (id_two.get(j) == 0) id_two[j] = 1;
  }

  // operators
  // +=
  id_assignment_tests(+=, +, id, id_two, id_three);

  // -=
  id_assignment_tests(-=, -, id, id_two, id_three);

  // *=
  id_assignment_tests(*=, *, id, id_two, id_three);

  // /=
  id_assignment_tests(/=, /, id, id_two, id_three);

  // %=
  id_assignment_tests(%=, %, id, id_two, id_three);

  // >>=
  id_assignment_tests(>>=, >>, id, id_two, id_three);

  // <<=
  id_assignment_tests(<<=, <<, id, id_two, id_three);

  // &=
  id_assignment_tests(&=, &, id, id_two, id_three);

  // |=
  id_assignment_tests(|=, |, id, id_two, id_three);

  // ^=
  id_assignment_tests(^=, ^, id, id_two, id_three);

  // *
  ID_KERNEL_TEST(*, id, id_two, id_three);

  // /
  ID_KERNEL_TEST(/, id, id_two, id_three);

  //+
  ID_KERNEL_TEST(+, id, id_two, id_three);

  //-
  ID_KERNEL_TEST(-, id, id_two, id_three);

  //%
  ID_KERNEL_TEST(%, id, id_two, id_three);

  //<<
  ID_KERNEL_TEST(<<, id, id_two, id_three);

  //>>
  ID_KERNEL_TEST(>>, id, id_two, id_three);

  //&
  ID_KERNEL_TEST(&, id, id_two, id_three);

  //|
  ID_KERNEL_TEST(|, id, id_two, id_three);

  //^
  ID_KERNEL_TEST (^, id, id_two, id_three);

  // ==
  ID_EQ_KERNEL_TEST(==, id, id_two);

  // !=
  ID_EQ_KERNEL_TEST(!=, id, id_two);

  // *
  DUAL_SIZE_ID_KERNEL_TEST(*, id, integer, id_three);

  // +
  DUAL_SIZE_ID_KERNEL_TEST(+, id, integer, id_three);

  // -
  DUAL_SIZE_ID_KERNEL_TEST(-, id, integer, id_three);

  // /
  DUAL_SIZE_ID_KERNEL_TEST(/, id, integer, id_three);

  // %
  DUAL_SIZE_ID_KERNEL_TEST(%, id, integer, id_three);

  // <<
  DUAL_SIZE_ID_KERNEL_TEST(<<, id, integer, id_three);

  // >>
  DUAL_SIZE_ID_KERNEL_TEST(>>, id, integer, id_three);

  // >
  if (!(id_two > id)) {
    error_ptr[m_iteration] = __LINE__;
    m_iteration++;
  }

  // <
  if (!(id < id_two)) {
    error_ptr[m_iteration] = __LINE__;
    m_iteration++;
  }

  // >=
  if (!(id_two > id)) {
    error_ptr[m_iteration] = __LINE__;
    m_iteration++;
  }

  // <=
  if (!(id <= id_two)) {
    error_ptr[m_iteration] = __LINE__;
    m_iteration++;
  }
}

template <int dims>
class test_id {
 public:
  // golden values
  static const int m_x = 16;
  static const int m_y = 32;
  static const int m_z = 64;
  static const int m_local = 2;
  static const int error_size =
      19 * 2;  // 19 test cases, with 2 variants each, worst case: all fail
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
          cl::sycl::id<dims> id(item);
          int check[] = {m_x, m_y, m_z};

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

/** test cl::sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      // use across all the dimensions
      auto my_queue = util::get_cts_object::queue();
      // templated approach
      {
        cl::sycl::range<1> range_1d_g(test_id<1>::m_x);
        cl::sycl::range<2> range_2d_g(test_id<2>::m_x, test_id<2>::m_y);
        cl::sycl::range<3> range_3d_g(test_id<3>::m_x, test_id<3>::m_y,
                                      test_id<3>::m_z);

        cl::sycl::range<1> range_1d_l(test_id<1>::m_local);
        cl::sycl::range<2> range_2d_l(test_id<2>::m_local, test_id<2>::m_local);
        cl::sycl::range<3> range_3d_l(test_id<3>::m_local, test_id<3>::m_local,
                                      test_id<3>::m_local);

        test_id<1> test1d;
        test1d(log, range_1d_g, range_1d_l, my_queue);
        test_id<2> test2d;
        test2d(log, range_2d_g, range_2d_l, my_queue);
        test_id<3> test3d;
        test3d(log, range_3d_g, range_3d_l, my_queue);
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace id_api__ */
