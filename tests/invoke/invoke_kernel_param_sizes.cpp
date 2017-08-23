/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME invoke_kernel_param_sizes

namespace invoke_kernel_param_sizes__ {
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T>
class type_size_kernel {
  typedef accessor<int32_t, 1, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      write_t;

  write_t m_out;

 public:
  type_size_kernel(write_t out_accessor) : m_out(out_accessor) {}

  void operator()() { m_out[0] = sizeof(T); }
};

template <typename T>
bool test_kernel_type_size(util::logger &log, queue &sycl_queue,
                           const sycl_cts::util::STRING &name) {
  using namespace sycl_cts::util;

  int32_t host_type_size = sizeof(T);
  int32_t kernel_type_size = 0;
  {
    cl::sycl::buffer<int32_t, 1> buffer_output(&kernel_type_size,
                                               cl::sycl::range<1>(1));
    sycl_queue.submit([&](handler &cgh) {
      auto access_output =
          buffer_output.template get_access<cl::sycl::access::mode::write>(cgh);
      type_size_kernel<T> kernel(access_output);
      cgh.single_task(kernel);
    });
  }

  if (host_type_size != kernel_type_size) {
    STRING msg = STRING("type size mismatch for: ") + STRING(name);
    msg += STRING("; device size = ") + std::to_string(kernel_type_size);
    msg += STRING(", host size = ") + std::to_string(host_type_size);
    return FAIL(log, msg);
  }

  return true;
}

/** test cl::sycl::kernel from functor
 */
class TEST_NAME : public sycl_cts::util::test_base {
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
      cts_selector selector;
      queue sycl_queue(selector);

      bool pass = true;

      // floating point types
      pass &= test_kernel_type_size<float>(log, sycl_queue, "float");
      pass &= test_kernel_type_size<double>(log, sycl_queue, "double");

      // unsigned scalar types
      pass &= test_kernel_type_size<uint8_t>(log, sycl_queue, "uint8_t");
      pass &= test_kernel_type_size<uint16_t>(log, sycl_queue, "uint16_t");
      pass &= test_kernel_type_size<uint32_t>(log, sycl_queue, "uint32_t");
      pass &= test_kernel_type_size<uint64_t>(log, sycl_queue, "uint64_t");

      // signed scalar types
      pass &= test_kernel_type_size<int8_t>(log, sycl_queue, "int8_t");
      pass &= test_kernel_type_size<int16_t>(log, sycl_queue, "int16_t");
      pass &= test_kernel_type_size<int32_t>(log, sycl_queue, "int32_t");
      pass &= test_kernel_type_size<int64_t>(log, sycl_queue, "int64_t");

      // floating point vector types
      pass &= test_kernel_type_size<float2>(log, sycl_queue, "float2");
      pass &= test_kernel_type_size<float3>(log, sycl_queue, "float3");
      pass &= test_kernel_type_size<float4>(log, sycl_queue, "float4");
      pass &= test_kernel_type_size<float8>(log, sycl_queue, "float8");
      pass &= test_kernel_type_size<float16>(log, sycl_queue, "float16");
      // double floating point vector types
      pass &= test_kernel_type_size<double2>(log, sycl_queue, "double2");
      pass &= test_kernel_type_size<double3>(log, sycl_queue, "double3");
      pass &= test_kernel_type_size<double4>(log, sycl_queue, "double4");
      pass &= test_kernel_type_size<double8>(log, sycl_queue, "double8");
      pass &= test_kernel_type_size<double16>(log, sycl_queue, "double16");

      // unsigned 1 byte vector types
      pass &= test_kernel_type_size<uchar2>(log, sycl_queue, "uchar2");
      pass &= test_kernel_type_size<uchar3>(log, sycl_queue, "uchar3");
      pass &= test_kernel_type_size<uchar4>(log, sycl_queue, "uchar4");
      pass &= test_kernel_type_size<uchar8>(log, sycl_queue, "uchar8");
      pass &= test_kernel_type_size<uchar16>(log, sycl_queue, "uchar16");
      // unsigned 2 byte vector types
      pass &= test_kernel_type_size<ushort2>(log, sycl_queue, "ushort2");
      pass &= test_kernel_type_size<ushort3>(log, sycl_queue, "ushort3");
      pass &= test_kernel_type_size<ushort4>(log, sycl_queue, "ushort4");
      pass &= test_kernel_type_size<ushort8>(log, sycl_queue, "ushort8");
      pass &= test_kernel_type_size<ushort16>(log, sycl_queue, "ushort16");
      // unsigned 4 byte vector types
      pass &= test_kernel_type_size<uint2>(log, sycl_queue, "uint2");
      pass &= test_kernel_type_size<uint3>(log, sycl_queue, "uint3");
      pass &= test_kernel_type_size<uint4>(log, sycl_queue, "uint4");
      pass &= test_kernel_type_size<uint8>(log, sycl_queue, "uint8");
      pass &= test_kernel_type_size<uint16>(log, sycl_queue, "uint16");
      // unsigned 8 byte vector types
      pass &= test_kernel_type_size<ulong2>(log, sycl_queue, "ulong2");
      pass &= test_kernel_type_size<ulong3>(log, sycl_queue, "ulong3");
      pass &= test_kernel_type_size<ulong4>(log, sycl_queue, "ulong4");
      pass &= test_kernel_type_size<ulong8>(log, sycl_queue, "ulong8");
      pass &= test_kernel_type_size<ulong16>(log, sycl_queue, "ulong16");

      // signed 1 byte vector types
      pass &= test_kernel_type_size<char2>(log, sycl_queue, "char2");
      pass &= test_kernel_type_size<char3>(log, sycl_queue, "char3");
      pass &= test_kernel_type_size<char4>(log, sycl_queue, "char4");
      pass &= test_kernel_type_size<char8>(log, sycl_queue, "char8");
      pass &= test_kernel_type_size<char16>(log, sycl_queue, "char16");
      // signed 2 byte vector types
      pass &= test_kernel_type_size<short2>(log, sycl_queue, "short2");
      pass &= test_kernel_type_size<short3>(log, sycl_queue, "short3");
      pass &= test_kernel_type_size<short4>(log, sycl_queue, "short4");
      pass &= test_kernel_type_size<short8>(log, sycl_queue, "short8");
      pass &= test_kernel_type_size<short16>(log, sycl_queue, "short16");
      // signed 4 byte vector types
      pass &= test_kernel_type_size<int2>(log, sycl_queue, "int2");
      pass &= test_kernel_type_size<int3>(log, sycl_queue, "int3");
      pass &= test_kernel_type_size<int4>(log, sycl_queue, "int4");
      pass &= test_kernel_type_size<int8>(log, sycl_queue, "int8");
      pass &= test_kernel_type_size<int16>(log, sycl_queue, "int16");
      // signed 8 byte vector types
      pass &= test_kernel_type_size<long2>(log, sycl_queue, "long2");
      pass &= test_kernel_type_size<long3>(log, sycl_queue, "long3");
      pass &= test_kernel_type_size<long4>(log, sycl_queue, "long4");
      pass &= test_kernel_type_size<long8>(log, sycl_queue, "long8");
      pass &= test_kernel_type_size<long16>(log, sycl_queue, "long16");

      if (!pass) FAIL(log, "one or more type size mismatches");

      sycl_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace invoke_kernel_param_sizes__ */
