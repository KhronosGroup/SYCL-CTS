/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME invoke_kernel_param_sizes

namespace invoke_kernel_param_sizes__ {
using namespace sycl_cts;

template <typename T>
class type_size_kernel {
  typedef cl::sycl::accessor<int32_t, 1, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer>
      write_t;

  write_t m_out;

 public:
  type_size_kernel(write_t out_accessor) : m_out(out_accessor) {}

  void operator()() { m_out[0] = sizeof(T); }
};

template <typename T>
bool test_kernel_type_size(util::logger &log, cl::sycl::queue &sycl_queue,
                           const cl::sycl::string_class &name) {
  int32_t host_type_size = sizeof(T);
  int32_t kernel_type_size = 0;
  {
    cl::sycl::buffer<int32_t, 1> buffer_output(&kernel_type_size,
                                               cl::sycl::range<1>(1));
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
      auto access_output =
          buffer_output.template get_access<cl::sycl::access::mode::write>(cgh);
      type_size_kernel<T> kernel(access_output);
      cgh.single_task(kernel);
    });
  }

  if (host_type_size != kernel_type_size) {
    cl::sycl::string_class msg =
        cl::sycl::string_class("type size mismatch for: ") +
        cl::sycl::string_class(name);
    msg += cl::sycl::string_class("; device size = ") +
           std::to_string(kernel_type_size);
    msg += cl::sycl::string_class(", host size = ") +
           std::to_string(host_type_size);
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
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      auto sycl_queue = util::get_cts_object::queue();

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
      pass &=
          test_kernel_type_size<cl::sycl::float2>(log, sycl_queue, "float2");
      pass &=
          test_kernel_type_size<cl::sycl::float3>(log, sycl_queue, "float3");
      pass &=
          test_kernel_type_size<cl::sycl::float4>(log, sycl_queue, "float4");
      pass &=
          test_kernel_type_size<cl::sycl::float8>(log, sycl_queue, "float8");
      pass &=
          test_kernel_type_size<cl::sycl::float16>(log, sycl_queue, "float16");
      // double floating point vector types
      pass &=
          test_kernel_type_size<cl::sycl::double2>(log, sycl_queue, "double2");
      pass &=
          test_kernel_type_size<cl::sycl::double3>(log, sycl_queue, "double3");
      pass &=
          test_kernel_type_size<cl::sycl::double4>(log, sycl_queue, "double4");
      pass &=
          test_kernel_type_size<cl::sycl::double8>(log, sycl_queue, "double8");
      pass &= test_kernel_type_size<cl::sycl::double16>(log, sycl_queue,
                                                        "double16");

      // unsigned 1 byte vector types
      pass &=
          test_kernel_type_size<cl::sycl::uchar2>(log, sycl_queue, "uchar2");
      pass &=
          test_kernel_type_size<cl::sycl::uchar3>(log, sycl_queue, "uchar3");
      pass &=
          test_kernel_type_size<cl::sycl::uchar4>(log, sycl_queue, "uchar4");
      pass &=
          test_kernel_type_size<cl::sycl::uchar8>(log, sycl_queue, "uchar8");
      pass &=
          test_kernel_type_size<cl::sycl::uchar16>(log, sycl_queue, "uchar16");
      // unsigned 2 byte vector types
      pass &=
          test_kernel_type_size<cl::sycl::ushort2>(log, sycl_queue, "ushort2");
      pass &=
          test_kernel_type_size<cl::sycl::ushort3>(log, sycl_queue, "ushort3");
      pass &=
          test_kernel_type_size<cl::sycl::ushort4>(log, sycl_queue, "ushort4");
      pass &=
          test_kernel_type_size<cl::sycl::ushort8>(log, sycl_queue, "ushort8");
      pass &= test_kernel_type_size<cl::sycl::ushort16>(log, sycl_queue,
                                                        "ushort16");
      // unsigned 4 byte vector types
      pass &= test_kernel_type_size<cl::sycl::uint2>(log, sycl_queue, "uint2");
      pass &= test_kernel_type_size<cl::sycl::uint3>(log, sycl_queue, "uint3");
      pass &= test_kernel_type_size<cl::sycl::uint4>(log, sycl_queue, "uint4");
      pass &= test_kernel_type_size<cl::sycl::uint8>(log, sycl_queue, "uint8");
      pass &=
          test_kernel_type_size<cl::sycl::uint16>(log, sycl_queue, "uint16");
      // unsigned 8 byte vector types
      pass &=
          test_kernel_type_size<cl::sycl::ulong2>(log, sycl_queue, "ulong2");
      pass &=
          test_kernel_type_size<cl::sycl::ulong3>(log, sycl_queue, "ulong3");
      pass &=
          test_kernel_type_size<cl::sycl::ulong4>(log, sycl_queue, "ulong4");
      pass &=
          test_kernel_type_size<cl::sycl::ulong8>(log, sycl_queue, "ulong8");
      pass &=
          test_kernel_type_size<cl::sycl::ulong16>(log, sycl_queue, "ulong16");

      // signed 1 byte vector types
      pass &= test_kernel_type_size<cl::sycl::char2>(log, sycl_queue, "char2");
      pass &= test_kernel_type_size<cl::sycl::char3>(log, sycl_queue, "char3");
      pass &= test_kernel_type_size<cl::sycl::char4>(log, sycl_queue, "char4");
      pass &= test_kernel_type_size<cl::sycl::char8>(log, sycl_queue, "char8");
      pass &=
          test_kernel_type_size<cl::sycl::char16>(log, sycl_queue, "char16");
      // signed 2 byte vector types
      pass &=
          test_kernel_type_size<cl::sycl::short2>(log, sycl_queue, "short2");
      pass &=
          test_kernel_type_size<cl::sycl::short3>(log, sycl_queue, "short3");
      pass &=
          test_kernel_type_size<cl::sycl::short4>(log, sycl_queue, "short4");
      pass &=
          test_kernel_type_size<cl::sycl::short8>(log, sycl_queue, "short8");
      pass &=
          test_kernel_type_size<cl::sycl::short16>(log, sycl_queue, "short16");
      // signed 4 byte vector types
      pass &= test_kernel_type_size<cl::sycl::int2>(log, sycl_queue, "int2");
      pass &= test_kernel_type_size<cl::sycl::int3>(log, sycl_queue, "int3");
      pass &= test_kernel_type_size<cl::sycl::int4>(log, sycl_queue, "int4");
      pass &= test_kernel_type_size<cl::sycl::int8>(log, sycl_queue, "int8");
      pass &= test_kernel_type_size<cl::sycl::int16>(log, sycl_queue, "int16");
      // signed 8 byte vector types
      pass &= test_kernel_type_size<cl::sycl::long2>(log, sycl_queue, "long2");
      pass &= test_kernel_type_size<cl::sycl::long3>(log, sycl_queue, "long3");
      pass &= test_kernel_type_size<cl::sycl::long4>(log, sycl_queue, "long4");
      pass &= test_kernel_type_size<cl::sycl::long8>(log, sycl_queue, "long8");
      pass &=
          test_kernel_type_size<cl::sycl::long16>(log, sycl_queue, "long16");

      if (!pass) FAIL(log, "one or more type size mismatches");

      sycl_queue.wait_and_throw();
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

} /* namespace invoke_kernel_param_sizes__ */
