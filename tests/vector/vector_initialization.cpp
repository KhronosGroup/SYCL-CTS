/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:  (c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "./../../util/math_helper.h"

using namespace cl::sycl;

#define TEST_NAME vector_initialization
#define KERNEL_NAME cKernel_vector_initialization

namespace vector_initalization__ {
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
    auto &o = m_o[item.get_global()];
    auto i = m_i[item.get_global()];
    o = i;
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
    MAKENAME(float);
    MAKENAME(float2);
    MAKENAME(float3);
    MAKENAME(float4);
    MAKENAME(float8);
    MAKENAME(float16);
#undef MAKENAME
    set_test_info(out, l_name, TEST_FILE);
  }

  /** execute this test
   *  @return, one of test_result enum
   */
  virtual void run(util::logger &log) {
    try {
      T idata;
      math::fill(idata, 1);
      T odata(0);

      // construct the cts default selector
      cts_selector selector;

      /* create command queue */
      queue l_queue(selector);
      {
        buffer<T, 1> ibuf(&idata, range<1>(math::numElements(idata)));
        buffer<T, 1> obuf(&odata, range<1>(math::numElements(odata)));

        int a = math::numElements(idata);

        /* add command to queue */
        l_queue.submit([&](handler &cgh) {
          auto iptr =
              ibuf.template get_access<cl::sycl::access::mode::read>(cgh);
          auto optr =
              obuf.template get_access<cl::sycl::access::mode::write>(cgh);

          /* instantiate the kernel */
          auto kern = KERNEL_NAME<T>(optr, iptr);

          /* execute the kernel */
          cgh.parallel_for(nd_range<1>(range<1>(1), range<1>(1)), kern);
        });
      }

      if (all(odata != idata)) {
        FAIL(log, "results don't match");
      }

      l_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME<float2>> proxy2;
util::test_proxy<TEST_NAME<float3>> proxy3;
util::test_proxy<TEST_NAME<float4>> proxy4;
util::test_proxy<TEST_NAME<float8>> proxy8;
util::test_proxy<TEST_NAME<float16>> proxy16;

}; /* vector_initalization__ */
