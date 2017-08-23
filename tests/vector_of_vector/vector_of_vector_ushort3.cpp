
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_of_vector.py
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

#define TEST_NAME vector_of_vector_ushort3
#define KERNEL_NAME cKernel_vector_of_vector
#define NUM_TESTS 1

/* CONSTRUCTOR TESTS SPECIAL MACROS FOR EASIER GENERATION AND VERIFICATION*/
#define V1S0 1
#define V2S0 2
#define V2S1 3
#define V3S0 4
#define V3S1 5
#define V3S2 6
#define V4S0 7
#define V4S1 8
#define V4S2 9
#define V4S3 10
#define V8S0 11
#define V8S1 12
#define V8S2 13
#define V8S3 14
#define V8S4 15
#define V8S5 16
#define V8S6 17
#define V8S7 18
#define V16S0 19
#define V16S1 20
#define V16S2 21
#define V16S3 22
#define V16S4 23
#define V16S5 24
#define V16S6 25
#define V16S7 26
#define V16S8 27
#define V16S9 28
#define V16S10 29
#define V16S11 30
#define V16S12 31
#define V16S13 32
#define V16S14 33
#define V16S15 34

#define V1 V1S0
#define V2 V2S0, V2S1
#define V3 V3S0, V3S1, V3S2
#define V4 V4S0, V4S1, V4S2, V4S3
#define V8 V8S0, V8S1, V8S2, V8S3, V8S4, V8S5, V8S6, V8S7
#define V16                                                             \
  V16S0, V16S1, V16S2, V16S3, V16S4, V16S5, V16S6, V16S7, V16S8, V16S9, \
      V16S10, V16S11, V16S12, V16S13, V16S14, V16S15

#define ushort1 ushort
#define CONSTRUCTOR_TEST(INDEX, ...) m_o[INDEX] = T(__VA_ARGS__);
#define VERIFY_EQUALS(INDEX, ...)            \
  if (all(odata[INDEX] != T(__VA_ARGS__))) { \
    FAIL(log, "results don't match");        \
  }

namespace TEST_NAME {
using namespace sycl_cts;

/** kernel functor
 */
template <typename T>
class KERNEL_NAME {
 protected:
  typedef accessor<T, 1, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      t_writeAccess;

  t_writeAccess m_o; /* output     */

 public:
  KERNEL_NAME(t_writeAccess out_) : m_o(out_) {}

  void operator()(item<1> item) {
    ushort1 v1(V1S0);
    ushort2 v2(V2S0, V2S1);
    ushort3 v3(V3S0, V3S1, V3S2);
    ushort4 v4(V4S0, V4S1, V4S2, V4S3);
    ushort8 v8(V8S0, V8S1, V8S2, V8S3, V8S4, V8S5, V8S6, V8S7);
    ushort16 v16(V16S0, V16S1, V16S2, V16S3, V16S4, V16S5, V16S6, V16S7, V16S8,
                 V16S9, V16S10, V16S11, V16S12, V16S13, V16S14, V16S15);

    /* MACROS GENERATED FROM PYTHON SCRIPT*/

    CONSTRUCTOR_TEST(0, v3)

    /* MACROS GENERATED FROM PYTHON SCRIPT*/
  }
};

/** test SYCL header for compilation
 */
template <typename T>
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  virtual void get_info(test_base::info &out) const {
    const char *l_name = "";
#define MAKENAME(X)                      \
  if (typeid(T) == typeid(X)) {          \
    l_name = TOSTRING(TEST_NAME) "_" #X; \
  }
    MAKENAME(ushort2);
    MAKENAME(ushort3);
    MAKENAME(ushort4);
    MAKENAME(ushort8);
    MAKENAME(ushort16);
#undef MAKENAME
    set_test_info(out, l_name, TEST_FILE);
  }

  /** execute this test
   *  @return, one of test_result enum
   */
  virtual void run(util::logger &log) {
    try {
      T odata[NUM_TESTS];

      // construct the cts default selector
      cts_selector selector;

      /* create command queue */
      queue l_queue(selector);

      buffer<T, 1> obuf(odata, range<1>(NUM_TESTS));

      /* add command to queue */
      l_queue.submit([&](handler &cgh) {
        auto optr =
            obuf.template get_access<cl::sycl::access::mode::write>(cgh);

        /* instantiate the kernel */
        auto kern = KERNEL_NAME<T>(optr);

        /* execute the kernel */
        cgh.parallel_for(nd_range<1>(range<1>(1), range<1>(1)), kern);
      });

      /* MACROS GENERATED FROM PYTHON SCRIPT*/

      VERIFY_EQUALS(0, V3)

      /* MACROS GENERATED FROM PYTHON SCRIPT*/

      l_queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME<ushort3>> proxy4;
}; /* vector_initalization__ */
