
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

#define TEST_NAME vector_swizzles_2
#define KERNEL_NAME cKernel_vector_swizzles
#define VECTOR_SIZE 2
#define NUM_TESTS 8

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

    RHS_SIMPLE_SWIZZLE(0, xy)
    RHS_SIMPLE_SWIZZLE(1, yx)

    LHS_SIMPLE_SWIZZLE(2, xy)
    LHS_SIMPLE_SWIZZLE(3, yx)

    RHS_TEMPLATE_SWIZZLE(4, X, Y)
    RHS_TEMPLATE_SWIZZLE(5, Y, X)

    LHS_TEMPLATE_SWIZZLE(6, X, Y)
    LHS_TEMPLATE_SWIZZLE(7, Y, X)

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

      SWIZZLE_VERIFY_EQUALS(0, X, Y)
      SWIZZLE_VERIFY_EQUALS(1, Y, X)

      SWIZZLE_VERIFY_EQUALS(2, X, Y)
      SWIZZLE_VERIFY_EQUALS(3, Y, X)

      SWIZZLE_VERIFY_EQUALS(4, X, Y)
      SWIZZLE_VERIFY_EQUALS(5, Y, X)

      SWIZZLE_VERIFY_EQUALS(6, X, Y)
      SWIZZLE_VERIFY_EQUALS(7, Y, X)

      /* MACROS GENERATED FROM PYTHON SCRIPT*/

      l_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(char, 2), char>> proxy2;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(uchar, 2), uchar>> proxy3;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(short, 2), short>> proxy4;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(ushort, 2), ushort>> proxy5;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(int, 2), int>> proxy6;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(uint, 2), uint>> proxy7;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(float, 2), float>> proxy8;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(double, 2), double>> proxy9;

}; /* vector_initalization__ */
