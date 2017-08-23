/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME vector_type_check

namespace vector_type_check__ {
using namespace sycl_cts;
using namespace cl::sycl;

template <typename X, typename Y>
class same_type {
 public:
  same_type(bool &pass) { pass &= false; }
};

template <typename X>
class same_type<X, X> {
 public:
  same_type(bool &pass) { pass &= true; }
};

#define SAME_TYPE(X, Y, Z)          \
  {                                 \
    auto e = X;                     \
    same_type<decltype(e), Y> v(Z); \
  }

typedef accessor<int32_t, 1, cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::global_buffer>
    pass_t;

template <typename TYPE, typename VEC>
class type_check_kernel_v2 {
 public:
  pass_t m_pass;

  type_check_kernel_v2(pass_t p) : m_pass(p) {}

  void operator()() {
    VEC vec;
    bool pass = true;

    SAME_TYPE(vec.x(), TYPE, pass);
    SAME_TYPE(vec.y(), TYPE, pass);
    m_pass[0] = pass;
  }
};

template <typename TYPE, typename VEC>
class type_check_kernel_v3 {
 public:
  pass_t m_pass;

  type_check_kernel_v3(pass_t p) : m_pass(p) {}

  void operator()() {
    VEC vec;
    bool pass = true;

    SAME_TYPE(vec.x(), TYPE, pass);
    SAME_TYPE(vec.y(), TYPE, pass);
    SAME_TYPE(vec.z(), TYPE, pass);
  }
};

template <typename TYPE, typename VEC>
class type_check_kernel_v4 {
 public:
  pass_t m_pass;

  type_check_kernel_v4(pass_t p) : m_pass(p) {}

  void operator()() {
    VEC vec;
    bool pass = true;

    SAME_TYPE(vec.x(), TYPE, pass);
    SAME_TYPE(vec.y(), TYPE, pass);
    SAME_TYPE(vec.z(), TYPE, pass);
    SAME_TYPE(vec.w(), TYPE, pass);
  }
};

template <typename TYPE, typename VEC>
class type_check_kernel_v8 {
 public:
  pass_t m_pass;

  type_check_kernel_v8(pass_t p) : m_pass(p) {}

  void operator()() {
    VEC vec;
    bool pass = true;

    SAME_TYPE(vec.s0(), TYPE, pass);
    SAME_TYPE(vec.s1(), TYPE, pass);
    SAME_TYPE(vec.s2(), TYPE, pass);
    SAME_TYPE(vec.s3(), TYPE, pass);
    SAME_TYPE(vec.s4(), TYPE, pass);
    SAME_TYPE(vec.s5(), TYPE, pass);
    SAME_TYPE(vec.s6(), TYPE, pass);
    SAME_TYPE(vec.s7(), TYPE, pass);
  }
};

template <typename TYPE, typename VEC>
class type_check_kernel_v16 {
 public:
  pass_t m_pass;

  type_check_kernel_v16(pass_t p) : m_pass(p) {}

  void operator()() {
    VEC vec;
    bool pass = true;

    SAME_TYPE(vec.s0(), TYPE, pass);
    SAME_TYPE(vec.s1(), TYPE, pass);
    SAME_TYPE(vec.s2(), TYPE, pass);
    SAME_TYPE(vec.s3(), TYPE, pass);
    SAME_TYPE(vec.s4(), TYPE, pass);
    SAME_TYPE(vec.s5(), TYPE, pass);
    SAME_TYPE(vec.s6(), TYPE, pass);
    SAME_TYPE(vec.s7(), TYPE, pass);
    SAME_TYPE(vec.s8(), TYPE, pass);
    SAME_TYPE(vec.s9(), TYPE, pass);
    SAME_TYPE(vec.sA(), TYPE, pass);
    SAME_TYPE(vec.sB(), TYPE, pass);
    SAME_TYPE(vec.sC(), TYPE, pass);
    SAME_TYPE(vec.sD(), TYPE, pass);
    SAME_TYPE(vec.sE(), TYPE, pass);
    SAME_TYPE(vec.sF(), TYPE, pass);
  }
};

template <typename KERNEL>
bool test_type(util::logger &log, queue &sycl_queue) {
  int32_t pass = 0;
  buffer<int32_t, 1> buf_pass(&pass, range<1>(1));

  using namespace sycl_cts::util;
  sycl_queue.submit([&](handler &cgh) {
    pass_t acc_pass =
        buf_pass.template get_access<cl::sycl::access::mode::read_write,
                                     cl::sycl::access::target::global_buffer>(
            cgh);
    KERNEL kernel(acc_pass);
    cgh.single_task<KERNEL>(kernel);
  });

  if (pass == 0) FAIL(log, "Data type error for vector");

  return pass == 1;
}

/** test cl::sycl::kernel from functor
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   *  @param out, test_base::info structure as output
   */
  virtual void get_info(test_base::info &out) const {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   *  @param log, test transcript logging class
   */
  virtual void run(util::logger &log) {
    try {
      cts_selector selector;
      queue sycl_queue(selector);

      test_type<type_check_kernel_v2<int8_t, char2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<int8_t, char3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<int8_t, char4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<int8_t, char8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<int8_t, char16>>(log, sycl_queue);

      test_type<type_check_kernel_v2<uint8_t, uchar2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<uint8_t, uchar3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<uint8_t, uchar4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<uint8_t, uchar8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<uint8_t, uchar16>>(log, sycl_queue);

      test_type<type_check_kernel_v2<int16_t, short2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<int16_t, short3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<int16_t, short4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<int16_t, short8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<int16_t, short16>>(log, sycl_queue);

      test_type<type_check_kernel_v2<uint16_t, ushort2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<uint16_t, ushort3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<uint16_t, ushort4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<uint16_t, ushort8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<uint16_t, ushort16>>(log, sycl_queue);

      test_type<type_check_kernel_v2<int32_t, int2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<int32_t, int3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<int32_t, int4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<int32_t, int8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<int32_t, int16>>(log, sycl_queue);

      test_type<type_check_kernel_v2<uint32_t, uint2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<uint32_t, uint3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<uint32_t, uint4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<uint32_t, uint8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<uint32_t, uint16>>(log, sycl_queue);

      test_type<type_check_kernel_v2<float, float2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<float, float3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<float, float4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<float, float8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<float, float16>>(log, sycl_queue);

      test_type<type_check_kernel_v2<double, double2>>(log, sycl_queue);
      test_type<type_check_kernel_v3<double, double3>>(log, sycl_queue);
      test_type<type_check_kernel_v4<double, double4>>(log, sycl_queue);
      test_type<type_check_kernel_v8<double, double8>>(log, sycl_queue);
      test_type<type_check_kernel_v16<double, double16>>(log, sycl_queue);

      sycl_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_type_sizes__ */
