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

#define TEST_NAME invoke_template_kernels

namespace invoke_template_kernels__ {
using namespace sycl_cts;

template <typename T>
class templated_functor {
  typedef cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer>
      read_t;
  typedef cl::sycl::accessor<T, 1, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer>
      write_t;

  read_t m_in;
  write_t m_out;

 public:
  templated_functor(read_t in, write_t out) : m_in(in), m_out(out) {}

  void operator()() { m_out[0] = m_in[0]; }
};

template <typename T>
bool test_kernel_functor(T in_value, util::logger &log,
                         cl::sycl::queue &sycl_queue) {
  T input = in_value, output = 0;
  {
    cl::sycl::buffer<T, 1> buffer_input(&input, cl::sycl::range<1>(1));
    cl::sycl::buffer<T, 1> buffer_output(&output, cl::sycl::range<1>(1));
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
      auto access_input =
          buffer_input.template get_access<cl::sycl::access::mode::read>(cgh);
      auto access_output =
          buffer_output.template get_access<cl::sycl::access::mode::write>(cgh);
      templated_functor<T> kernel(access_input, access_output);
      cgh.single_task(kernel);
    });
  }
  return CHECK_VALUE(log, input, output, 0);
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

      static const float test_float_value = 10;
      static const double test_double_value = 10;

      if (!test_kernel_functor(test_float_value, log, sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(test_double_value, log, sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<int8_t>(INT8_MAX), log,
                               sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<int16_t>(INT16_MAX), log,
                               sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<int32_t>(INT32_MAX), log,
                               sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<int64_t>(INT64_MAX), log,
                               sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<uint8_t>(UINT8_MAX), log,
                               sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<uint16_t>(UINT16_MAX), log,
                               sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<uint32_t>(UINT32_MAX), log,
                               sycl_queue)) {
        return;
      }

      if (!test_kernel_functor(static_cast<uint64_t>(UINT64_MAX), log,
                               sycl_queue)) {
        return;
      }

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

} /* namespace invoke_template_kernels__ */
