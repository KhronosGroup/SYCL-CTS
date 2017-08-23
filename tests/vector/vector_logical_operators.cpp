/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:  (c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "./../../util/math_helper.h"

#define TEST_NAME vector_logical_operators
#define KERNEL_NAME cKernel_vector_logical_operators

namespace vector_logical_operators__ {
using namespace cl::sycl;
using namespace sycl_cts;

template <typename T>
class KERNEL_NAME {
 protected:
  typedef accessor<int, 1, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      t_writeAccess;
  typedef accessor<T, 1, cl::sycl::access::mode::read,
                   cl::sycl::access::target::global_buffer>
      t_readAccess;

  t_writeAccess m_o; /* output     */
  t_readAccess m_x;  /* argument X */
  t_readAccess m_y;  /* argument Y */

 public:
  KERNEL_NAME(t_writeAccess out_, t_readAccess x_, t_readAccess y_)
      : m_o(out_), m_x(x_), m_y(y_) {}

  void operator()() {
    T &x = m_x[0];
    T &y = m_y[0];

    if (all(x == y)) {
      m_o[0] = 0;
    } else {
      m_o[0] = 1;
    }

    if (all(x != y)) {
      m_o[1] = 1;
    } else {
      m_o[1] = 0;
    }
  }
};

/** test SYCL header for compilation
 */
template <typename T>
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const {
    using sycl_cts::util::STRING;
    STRING name = STRING(TOSTRING(TEST_NAME)) + "_" + type_name<T>();
    set_test_info(out, name, TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) {
    try {
      T lhs_data, rhs_data;
      int output[2] = {-1, -1};

      math::fill(lhs_data, 1);
      math::fill(rhs_data, 0);

      default_selector selector;
      queue sycl_queue(selector);

      {
        buffer<int, 1> buf_o(output, range<1>(2));
        buffer<T, 1> buf_x(&lhs_data, range<1>(1));
        buffer<T, 1> buf_y(&rhs_data, range<1>(1));

        sycl_queue.submit([&](handler &cgh) {
          auto acc_o =
              buf_o.template get_access<cl::sycl::access::mode::write>(cgh);
          auto acc_x =
              buf_x.template get_access<cl::sycl::access::mode::read>(cgh);
          auto acc_y =
              buf_y.template get_access<cl::sycl::access::mode::read>(cgh);

          KERNEL_NAME<T> run_kernel(acc_o, acc_x, acc_y);

          cgh.single_task(run_kernel);
        });
      }

      if (output[0] != 1) {
        FAIL(log, "logical operator == NOT functioning properly");
        return;
      }

      if (output[1] != 1) {
        FAIL(log, "logical operator != NOT functioning properly");
        return;
      }

      sycl_queue.wait_and_throw();

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

}; /* namespace vector_logical_operators__ */
