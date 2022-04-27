/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_INVOKE_TEMPLATE_KERNELS_COMMON_H
#define __SYCLCTS_TESTS_INVOKE_TEMPLATE_KERNELS_COMMON_H

#include "../common/common.h"

namespace invoke_template_kernels_common {
using namespace sycl_cts;

template <typename T>
class templated_functor {
  typedef sycl::accessor<T, 1, sycl::access_mode::read, sycl::target::device>
      read_t;
  typedef sycl::accessor<T, 1, sycl::access_mode::write, sycl::target::device>
      write_t;

  read_t m_in;
  write_t m_out;

 public:
  templated_functor(read_t in, write_t out) : m_in(in), m_out(out) {}

  void operator()() const { m_out[0] = m_in[0]; }
};

template <typename T>
bool test_kernel_functor(T in_value, sycl::queue &sycl_queue) {
  T input = in_value, output = 0;
  {
    sycl::buffer<T, 1> buffer_input(&input, sycl::range<1>(1));
    sycl::buffer<T, 1> buffer_output(&output, sycl::range<1>(1));
    sycl_queue.submit([&](sycl::handler &cgh) {
      auto access_input =
          buffer_input.template get_access<sycl::access_mode::read>(cgh);
      auto access_output =
          buffer_output.template get_access<sycl::access_mode::write>(cgh);
      templated_functor<T> kernel(access_input, access_output);
      cgh.single_task(kernel);
    });
  }
  return input == output;
}
} /* namespace invoke_template_kernels_common */
#endif  // __SYCLCTS_TESTS_INVOKE_TEMPLATE_KERNELS_COMMON_H
