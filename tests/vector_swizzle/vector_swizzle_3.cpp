
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_swizzle.py
//
************************************************************************************/
/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:  (c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#define SYCL_SIMPLE_SWIZZLES
#include "../common/common.h"
#include "./../../util/math_helper.h"

using namespace cl::sycl;

#define TEST_NAME vector_swizzles_3
#define KERNEL_NAME cKernel_vector_swizzles
#define VECTOR_SIZE 3
#define NUM_TESTS 24

/* SWIZZLE TESTS SPECIAL MACROS FOR EASIER GENERATION AND VERIFICATION*/
#define X 0
#define Y 1
#define Z 2
#define W 3
#define CREATE_VECTOR_TYPE(TYPE, SIZE) TYPE##SIZE
#define RHS_SIMPLE_SWIZZLE(INDEX, VARIATION) \
  m_o[INDEX] = m_i[INDEX].VARIATION();
#define LHS_SIMPLE_SWIZZLE(INDEX, VARIATION) \
  m_o[INDEX].VARIATION() = m_i[INDEX];
#define RHS_TEMPLATE_SWIZZLE(INDEX, ...) \
  m_o[INDEX] = m_i[INDEX].template swizzle<__VA_ARGS__>();
#define LHS_TEMPLATE_SWIZZLE(INDEX, ...) \
  m_o[INDEX].template swizzle<__VA_ARGS__>() = m_i[INDEX];
#define SWIZZLE_VERIFY_EQUALS(INDEX, ...)    \
  if (all(odata[INDEX] != T(__VA_ARGS__))) { \
    FAIL(log, "results don't match");        \
  }

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** kernel functor
 */
template <typename T>
class KERNEL_NAME {
 protected:
  typedef accessor<T, 1, cl::sycl::access::mode::read,
                   cl::sycl::access::target::global_buffer>
      t_readAccess;
  typedef accessor<T, 1, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      t_writeAccess;

  t_writeAccess m_o; /* output     */
  t_readAccess m_i;  /* input */

 public:
  KERNEL_NAME(t_writeAccess out_, t_readAccess in_) : m_o(out_), m_i(in_) {}

  void operator()(item<1> item) {
    /* MACROS GENERATED FROM PYTHON SCRIPT*/

    RHS_SIMPLE_SWIZZLE(0, xyz)
    RHS_SIMPLE_SWIZZLE(1, xzy)
    RHS_SIMPLE_SWIZZLE(2, yxz)
    RHS_SIMPLE_SWIZZLE(3, yzx)
    RHS_SIMPLE_SWIZZLE(4, zxy)
    RHS_SIMPLE_SWIZZLE(5, zyx)

    LHS_SIMPLE_SWIZZLE(6, xyz)
    LHS_SIMPLE_SWIZZLE(7, xzy)
    LHS_SIMPLE_SWIZZLE(8, yxz)
    LHS_SIMPLE_SWIZZLE(9, yzx)
    LHS_SIMPLE_SWIZZLE(10, zxy)
    LHS_SIMPLE_SWIZZLE(11, zyx)

    RHS_TEMPLATE_SWIZZLE(12, X, Y, Z)
    RHS_TEMPLATE_SWIZZLE(13, X, Z, Y)
    RHS_TEMPLATE_SWIZZLE(14, Y, X, Z)
    RHS_TEMPLATE_SWIZZLE(15, Y, Z, X)
    RHS_TEMPLATE_SWIZZLE(16, Z, X, Y)
    RHS_TEMPLATE_SWIZZLE(17, Z, Y, X)

    LHS_TEMPLATE_SWIZZLE(18, X, Y, Z)
    LHS_TEMPLATE_SWIZZLE(19, X, Z, Y)
    LHS_TEMPLATE_SWIZZLE(20, Y, X, Z)
    LHS_TEMPLATE_SWIZZLE(21, Y, Z, X)
    LHS_TEMPLATE_SWIZZLE(22, Z, X, Y)
    LHS_TEMPLATE_SWIZZLE(23, Z, Y, X)

    /* MACROS GENERATED FROM PYTHON SCRIPT*/
  }
};

/** test SYCL header for compilation
 */
template <typename T, typename E>
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    using sycl_cts::util::STRING;
    STRING name = STRING(TOSTRING(TEST_NAME) + STRING("_") + type_name<T>());
    set_test_info(out, name.c_str(), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) {
    try {
      E vector_input_data[4] = {X, Y, Z, W};
      T idata[NUM_TESTS];
      T odata[NUM_TESTS];

      for (int i = 0; i < NUM_TESTS; i++) {
        for (int j = 0; j < VECTOR_SIZE; j++) {
          math::setElement(idata[i], j, vector_input_data[j]);
        }
      }

      // construct the cts default selector
      cts_selector selector;

      /* create command queue */
      queue l_queue(selector);

      buffer<T, 1> ibuf(idata, range<1>(NUM_TESTS));
      buffer<T, 1> obuf(odata, range<1>(NUM_TESTS));

      /* add command to queue */
      l_queue.submit([&](handler &cgh) {
        auto iptr = ibuf.template get_access<cl::sycl::access::mode::read>(cgh);
        auto optr =
            obuf.template get_access<cl::sycl::access::mode::write>(cgh);

        /* instantiate the kernel */
        auto kern = KERNEL_NAME<T>(optr, iptr);

        /* execute the kernel */
        cgh.parallel_for(nd_range<1>(range<1>(1), range<1>(1)), kern);
      });

      /* MACROS GENERATED FROM PYTHON SCRIPT*/

      SWIZZLE_VERIFY_EQUALS(0, X, Y, Z)
      SWIZZLE_VERIFY_EQUALS(1, X, Z, Y)
      SWIZZLE_VERIFY_EQUALS(2, Y, X, Z)
      SWIZZLE_VERIFY_EQUALS(3, Y, Z, X)
      SWIZZLE_VERIFY_EQUALS(4, Z, X, Y)
      SWIZZLE_VERIFY_EQUALS(5, Z, Y, X)

      SWIZZLE_VERIFY_EQUALS(6, X, Y, Z)
      SWIZZLE_VERIFY_EQUALS(7, X, Z, Y)
      SWIZZLE_VERIFY_EQUALS(8, Y, X, Z)
      SWIZZLE_VERIFY_EQUALS(9, Y, Z, X)
      SWIZZLE_VERIFY_EQUALS(10, Z, X, Y)
      SWIZZLE_VERIFY_EQUALS(11, Z, Y, X)

      SWIZZLE_VERIFY_EQUALS(12, X, Y, Z)
      SWIZZLE_VERIFY_EQUALS(13, X, Z, Y)
      SWIZZLE_VERIFY_EQUALS(14, Y, X, Z)
      SWIZZLE_VERIFY_EQUALS(15, Y, Z, X)
      SWIZZLE_VERIFY_EQUALS(16, Z, X, Y)
      SWIZZLE_VERIFY_EQUALS(17, Z, Y, X)

      SWIZZLE_VERIFY_EQUALS(18, X, Y, Z)
      SWIZZLE_VERIFY_EQUALS(19, X, Z, Y)
      SWIZZLE_VERIFY_EQUALS(20, Y, X, Z)
      SWIZZLE_VERIFY_EQUALS(21, Y, Z, X)
      SWIZZLE_VERIFY_EQUALS(22, Z, X, Y)
      SWIZZLE_VERIFY_EQUALS(23, Z, Y, X)

      /* MACROS GENERATED FROM PYTHON SCRIPT*/

      l_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(char, 3), char>> proxy2;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(uchar, 3), uchar>> proxy3;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(short, 3), short>> proxy4;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(ushort, 3), ushort>> proxy5;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(int, 3), int>> proxy6;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(uint, 3), uint>> proxy7;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(float, 3), float>> proxy8;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(double, 3), double>> proxy9;

}; /* vector_initalization__ */
