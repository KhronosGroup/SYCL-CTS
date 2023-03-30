/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#define TEST_NAME range_api

namespace range_api__ {
using namespace sycl_cts;

template <int dims>
class test_range_kernel {};

template <int dims>
void test_range_kernels(
    sycl::range<dims> range,
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::device>
        error_ptr,
    int m_iteration) {
  // scratch value to test the operators
  sycl::range<dims> result(range);

  // non-zero values to test the operators
  size_t integer = 16;
  sycl::range<dims> range_two(range * 2);
  for (int j = 0; j < dims; j++) {
    if (range_two.get(j) == 0) {
      range_two[j] = 1;
    }
  }

  const sycl::range<dims> range_two_const(range_two);
  const sycl::range<dims> range_const(range);

  // friend bool operator==(const T& lhs, const T& rhs)
  INDEX_EQ_KERNEL_TEST(==, range, range_two);

  // friend bool operator!=(const T& lhs, const T& rhs)
  INDEX_EQ_KERNEL_TEST(!=, range, range_two);

  // friend range operatorOP(const range& lhs, const range& rhs)
  INDEX_KERNEL_TEST(+, range, range_two_const, result);
  INDEX_KERNEL_TEST(-, range, range_two_const, result);
  INDEX_KERNEL_TEST(*, range, range_two_const, result);
  INDEX_KERNEL_TEST(/, range, range_two_const, result);
  INDEX_KERNEL_TEST(%, range, range_two_const, result);
  INDEX_KERNEL_TEST(<<, range, range_two_const, result);
  INDEX_KERNEL_TEST(>>, range, range_two_const, result);
  INDEX_KERNEL_TEST(&, range, range_two_const, result);
  INDEX_KERNEL_TEST(|, range, range_two_const, result);
  INDEX_KERNEL_TEST(^, range, range_two_const, result);
  INDEX_KERNEL_TEST(&&, range, range_two_const, result);
  INDEX_KERNEL_TEST(||, range, range_two_const, result);
  INDEX_KERNEL_TEST(<, range, range_two_const, result);
  INDEX_KERNEL_TEST(>, range, range_two_const, result);
  INDEX_KERNEL_TEST(<=, range, range_two_const, result);
  INDEX_KERNEL_TEST(>=, range, range_two_const, result);

  // friend range operatorOP(const range& lhs, const size_t& rhs)
  // friend range operatorOP(const size_t& lhs, const range& rhs)
  DUAL_SIZE_INDEX_KERNEL_TEST(+, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(-, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(*, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(/, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(%, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(<<, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(>>, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(&, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(|, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(^, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(&&, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(||, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(<, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(>, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(<=, range, integer, result);
  DUAL_SIZE_INDEX_KERNEL_TEST(>=, range, integer, result);

  // friend range& operatorOP(range& lhs, const range& rhs)
  INDEX_ASSIGNMENT_TESTS(+=, +, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(-=, -, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(*=, *, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(/=, /, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(%=, %, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(<<=, <<, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(>>=, >>, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(&=, &, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(|=, |, range, range_two, result);
  INDEX_ASSIGNMENT_TESTS(^=, ^, range, range_two, result);

  // friend range& operatorOP(range& lhs, const size_t& rhs)
  INDEX_ASSIGNMENT_INTEGER_TESTS(+=, +, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(-=, -, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(*=, *, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(/=, /, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(%=, %, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(<<=, <<, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(>>=, >>, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(&=, &, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(|=, |, range, integer, result);
  INDEX_ASSIGNMENT_INTEGER_TESTS(^=, ^, range, integer, result);

  // friend range operatorOP(const range& rhs)
  UNARY_INDEX_KERNEL_TEST(+, range, result);
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
  UNARY_INDEX_KERNEL_TEST(-, range, result);
#endif

  // friend range& operatorOP(range& rhs)
  PREFIX_INDEX_KERNEL_TEST(++, range, result);
  PREFIX_INDEX_KERNEL_TEST(--, range, result);

  // friend range operatorOP(range& lhs, int)
  POSTFIX_INDEX_KERNEL_TEST(++, range, result);
  POSTFIX_INDEX_KERNEL_TEST(--, range, result);
}

template <int dims>
class test_range {
 public:
  // golden values
  static const int m_x = 16;
  static const int m_y = 32;
  static const int m_z = 64;
  static const int m_local = 2;
  // maximum amount of errors that the kernel can produce
  static const int error_size =
      96 * 3 + 2 +  // for operator checks, 3 from maximum dimension
      4;            // from other checks
  int m_error[error_size];

  void operator()(util::logger &log, sycl::range<dims> global,
                  sycl::range<dims> local, sycl::queue q) {
    // for testing get()
    for (int i = 0; i < error_size; i++) {
      m_error[i] = 0;  // no error
    }

    {
      sycl::buffer<int, 1> error_buffer(m_error, sycl::range<1>(error_size));

      q.submit([&](sycl::handler &cgh) {
        auto my_range = sycl::nd_range<dims>(global, local);

        auto error_ptr =
            error_buffer.get_access<sycl::access_mode::read_write>(cgh);

        auto my_kernel = ([=](sycl::nd_item<dims> item) {
          int m_iteration = 0;

          // create check table
          sycl::range<dims> range = item.get_nd_range().get_global_range();

          size_t check[] = {m_x, m_y, m_z};

          if (dims == 1) {
            if (range.size() != m_x) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          } else if (dims == 2) {
            if (range.size() != m_x * m_y) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          } else if (dims == 3) {
            if (range.size() != m_x * m_y * m_z) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          }

          for (int i = 0; i < dims; i++) {
            if (range.get(i) > check[i] || range[i] > check[i]) {
              // report an error
              error_ptr[m_iteration] = __LINE__;
              m_iteration++;
            }
          }

          test_range_kernels<dims>(range, error_ptr,
                                   m_iteration);  // test all in the kernel
        });
        cgh.parallel_for<class test_range_kernel<dims>>(my_range, my_kernel);
      });

      q.wait_and_throw();
    }
    for (int i = 0; i < error_size; i++) {
      CHECK_VALUE(log, m_error[i], 0, i);
    }
  }
};

void test_empty_kernel_1d_range(size_t N, sycl::queue q) {
    int *a = sycl::malloc_shared<int>(q, 1);
    a[0] = 0;
    q.parallel_for(N, [=](auto i){ a[0] = 1 ; }).wait_and_throw();
    CHECK_VALUE_SCALAR(log, a[0], 1);
    sycl::free(a, q);
}

/** test sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    {
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
      WARN(
          "ComputeCpp does not implement unary minus operation. "
          "Skipping the test for this operation.");
#endif

      // use across all the dimensions
      auto my_queue = util::get_cts_object::queue();
      // templated approach
      {
        sycl::range<1> range_1d_g(test_range<1>::m_x);
        sycl::range<2> range_2d_g(test_range<2>::m_x, test_range<2>::m_y);
        sycl::range<3> range_3d_g(test_range<3>::m_x, test_range<3>::m_y,
                                  test_range<3>::m_z);

        sycl::range<1> range_1d_l(test_range<1>::m_local);
        sycl::range<2> range_2d_l(test_range<2>::m_local,
                                  test_range<2>::m_local);
        sycl::range<3> range_3d_l(test_range<3>::m_local,
                                  test_range<3>::m_local,
                                  test_range<3>::m_local);

        test_range<1> test1d;
        test1d(log, range_1d_g, range_1d_l, my_queue);
        test_range<2> test2d;
        test2d(log, range_2d_g, range_2d_l, my_queue);
        test_range<3> test3d;
        test3d(log, range_3d_g, range_3d_l, my_queue);

        // 32 bits, it's trivial. Sanity test.
        test_empty_kernel_1d_range(std::numeric_limits<unsigned int>::max(), my_queue);
        // Most GPU hardware have limitation to 32bits * 1024 for their native API
        // so let's try more than that
        test_empty_kernel_1d_range(std::numeric_limits<unsigned int>::max()*2024L, my_queue);
      }
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace range_api__ */
