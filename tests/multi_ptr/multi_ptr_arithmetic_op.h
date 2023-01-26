/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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

//  Provides code for multi_ptr arithmetic operators

#ifndef __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H
#define __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

#include <optional>  // for std::optional

namespace multi_ptr_arithmetic_op {

namespace detail {

/**
 * @brief Structure that combines variables used to validate test results
 * @tparam T Data type that will be used in test
 */
template <typename T>
struct test_results {
  // Use std::optional to disable verification if some member hasn't been
  // initialised by some value

  // Expected value that will be returned after multi_ptr's operator will be
  // called
  std::optional<T> op_return_result_val;
  // Expected value that will be stored into multi_ptr after
  // multi_ptr's operator will be called
  std::optional<T> op_calling_val;
};

}  // namespace detail

template <typename T, typename AddrSpaceT, typename IsDecoratedT,
          typename KernelName>
class kernel_arithmetic_op;

/**
 * @brief Provides functor for verification on multi_ptr arithmetic operators
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_multi_ptr_arithmetic_op_test {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  using multi_ptr_t = sycl::multi_ptr<T, space, decorated>;

  static constexpr size_t m_array_size = 10;
  // Array that will be used in multi_ptr
  std::array<T, m_array_size> m_arr =
      multi_ptr_common::init_array<T, m_array_size>::value;
  static constexpr size_t m_middle_elem_index = m_array_size / 2;

  template <typename KernelName, typename TestActionT>
  void run_test(sycl::queue &queue, TestActionT test_action,
                detail::test_results<T> &expected_results) {
    // Variable that contains all variables that will be used to verify test
    // result
    detail::test_results<T> test_results;

    {
      sycl::range m_r(1);
      sycl::buffer<detail::test_results<T>> test_result_buffer(&test_results,
                                                               m_r);

      sycl::buffer<T> arr_buffer(m_arr.data(), sycl::range(m_array_size));
      queue.submit([&](sycl::handler &cgh) {
        using kname =
            kernel_arithmetic_op<T, AddrSpaceT, IsDecoratedT, KernelName>;
        auto test_result_acc =
            test_result_buffer.template get_access<sycl::access_mode::write>(
                cgh);
        auto arr_acc =
            arr_buffer.template get_access<sycl::access_mode::read>(cgh);

        if constexpr (space == sycl::access::address_space::local_space) {
          sycl::local_accessor<T, 1> acc_for_mptr{sycl::range(m_array_size),
                                                  cgh};
          cgh.parallel_for<kname>(
              sycl::nd_range(m_r, m_r), [=](sycl::nd_item<1> item) {
                for (size_t i = 0; i < m_array_size; ++i)
                  value_operations::assign(acc_for_mptr[i], arr_acc[i]);
                sycl::group_barrier(item.get_group());
                test_action(acc_for_mptr, test_result_acc);
              });
        } else if constexpr (space ==
                             sycl::access::address_space::private_space) {
          cgh.single_task<kname>([=] {
            std::array<T, m_array_size> priv_arr(
                multi_ptr_common::init_array<T, m_array_size>::value);
            for (size_t i = 0; i < m_array_size; ++i)
              value_operations::assign(priv_arr[i], arr_acc[i]);
            sycl::multi_ptr<T, sycl::access::address_space::private_space,
                            decorated>
                priv_arr_mptr = sycl::address_space_cast<
                    sycl::access::address_space::private_space, decorated>(
                    priv_arr.data());
            test_action(priv_arr_mptr, test_result_acc);
          });
        } else {
          cgh.single_task<kname>(
              [=] { test_action(arr_acc, test_result_acc); });
        }
      });
    }
    // If expected value isn't initialized, then this verification should be
    // skipped
    if (expected_results.op_return_result_val) {
      CHECK(expected_results.op_return_result_val ==
            test_results.op_return_result_val);
    }
    // If expected value isn't initialized, then this verification should be
    // skipped
    if (expected_results.op_calling_val) {
      CHECK(expected_results.op_calling_val == test_results.op_calling_val);
    }
  }

 public:
  /**
   * @param type_name Current data type string representation
   * @param address_space_name Current sycl::access::address_space string
   *        representation
   * @param is_decorated_name Current sycl::access::decorated string
   *        representation
   */
  void operator()(const std::string &type_name,
                  const std::string &address_space_name,
                  const std::string &is_decorated_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();

    for (size_t i = 0; i < m_array_size; ++i) {
      m_arr[i] = i;
    }

    SECTION(sycl_cts::section_name("Check multi_ptr operator++(multi_ptr& mp)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that will be returned after calling multi_ptr's operator
      verification_points.op_return_result_val = m_arr[1];
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_calling_val = m_arr[1];

      const auto run_test_action = [](auto acc_for_mptr, auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        multi_ptr_t result_mptr = ++mptr;

        detail::test_results<T> &test_result = test_result_acc[0];

        test_result.op_return_result_val = *result_mptr;
        test_result.op_calling_val = *mptr;
      };
      run_test<class pre_inc>(queue, run_test_action, verification_points);
    }
    SECTION(
        sycl_cts::section_name("Check multi_ptr operator++(multi_ptr&, int)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that will be returned after calling multi_ptr's operator
      verification_points.op_return_result_val = m_arr[0];
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_calling_val = m_arr[1];

      const auto run_test_action = [](auto acc_for_mptr, auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        multi_ptr_t result_mptr = mptr++;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *result_mptr;
        test_result.op_calling_val = *mptr;
      };
      run_test<class post_inc>(queue, run_test_action, verification_points);
    }
    SECTION(sycl_cts::section_name("Check multi_ptr operator--(multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that will be returned after calling multi_ptr's operator
      verification_points.op_return_result_val = m_arr[m_middle_elem_index - 1];
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_calling_val = m_arr[m_middle_elem_index - 1];

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        // Shift multi_ptr that he is pointed to the middle element, to have
        // possibe decrease pointed value index
        mptr += m_middle_elem_index;

        multi_ptr_t result_mptr = --mptr;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *result_mptr;
        test_result.op_calling_val = *mptr;
      };
      run_test<class pre_dec>(queue, run_test_action, verification_points);
    }
    SECTION(
        sycl_cts::section_name("Check multi_ptr operator--(multi_ptr&, int)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that will be returned after calling multi_ptr's operator
      verification_points.op_return_result_val = m_arr[m_middle_elem_index];
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_calling_val = m_arr[m_middle_elem_index - 1];

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        // Shift multi_ptr that he is pointed to the middle element, to have
        // possibe decrease pointed value index
        mptr += m_middle_elem_index;

        multi_ptr_t result_mptr = mptr--;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *result_mptr;
        test_result.op_calling_val = *mptr;
      };
      run_test<class post_dec>(queue, run_test_action, verification_points);
    }

    using diff_t = typename multi_ptr_t::difference_type;
    diff_t shift = m_array_size / 3;
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator+=(multi_ptr&, diff_type)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_calling_val = m_arr[shift];

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        mptr += shift;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_calling_val = *mptr;
      };
      run_test<class add_assign>(queue, run_test_action, verification_points);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator-=(multi_ptr&, diff_type)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_calling_val = m_arr[m_middle_elem_index - shift];

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        // Shift multi_ptr that he is pointed to the middle element, to have
        // possibe decrease pointed value index
        mptr += m_middle_elem_index;

        mptr -= shift;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_calling_val = *mptr;
      };
      run_test<class sub_assign>(queue, run_test_action, verification_points);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator+(const multi_ptr&, diff_type)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_return_result_val = m_arr[shift];

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        const multi_ptr_t mptr(acc_for_mptr);
        multi_ptr_t result_mptr = mptr + shift;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *result_mptr;
      };
      run_test<class add>(queue, run_test_action, verification_points);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator-(const multi_ptr&, diff_type)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_return_result_val =
          m_arr[m_middle_elem_index - shift];

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        // Shift multi_ptr that he is pointed to the middle element, to have
        // possibe decrease pointed value index
        mptr += m_middle_elem_index;

        multi_ptr_t result_mptr = mptr - shift;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *result_mptr;
      };
      run_test<class sub>(queue, run_test_action, verification_points);
    }
    SECTION(
        sycl_cts::section_name("Check multi_ptr operator*(const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::test_results<T> verification_points;
      // Expected value that be contained into multi_ptr after multi_ptr's
      // operator will be called
      verification_points.op_return_result_val = m_arr[0];

      const auto run_test_action = [](auto acc_for_mptr, auto test_result_acc) {
        const multi_ptr_t mptr(acc_for_mptr);

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *mptr;
      };
      run_test<class deref>(queue, run_test_action, verification_points);
    }
  }
};

template <typename T>
class check_multi_ptr_arithmetic_op_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();
    // Run test
    for_all_combinations<run_multi_ptr_arithmetic_op_test, T>(
        address_spaces, is_decorated, type_name);
  }
};

}  // namespace multi_ptr_arithmetic_op

#endif  // __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H
