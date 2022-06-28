/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for multi_ptr arithmetic operators
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H
#define __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_arithmetic_op {

namespace detail {

/**
 * @brief Structure that combines variables used to validate test results
 * @tparam T Data type that will be used in test
 */
template <typename T>
struct test_results {
  // Value that will be used to initialze variables with random values to avoid
  // false positive test result
  constexpr int value_to_init = 49;
  // Expected value that will be returned after multi_ptr's operator will be
  // called
  T op_return_result_val = value_to_init;
  // Expected value that will be stored into multi_ptr after
  // multi_ptr's operator will be called
  T op_calling_val = value_to_init;
};

}  // namespace detail

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

  constexpr size_t m_array_size = 10;
  // Array that will be used in multi_ptr
  T m_arr[m_array_size];
  size_t m_middle_elem_index = m_array_size / 2;

  sycl::range m_r = sycl::range(1);
  T m_default_expected_val = 55;

  template <typename TestActionT>
  void run_test(sycl::queue &queue, TestActionT test_action,
                detail::test_results<T> &expected_results) {
    // Variable that contains all variables that will be used to verify test
    // result
    detail::test_results<T> test_results;

    {
      sycl::buffer<detail::test_results<T>> test_result_buffer(&test_results,
                                                               sycl::range(1));

      sycl::buffer<T> buffer_for_mptr(m_arr, sycl::range(m_array_size));
      queue.submit([&](sycl::handler &cgh) {
        auto test_result_acc =
            test_result_buffer.template get_access<sycl::access_mode::write>(
                cgh);
        auto acc_for_mptr =
            buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);

        if constexpr (space == sycl::access::address_space::global_space) {
          cgh.single_task([=] { test_action(acc_for_mptr, test_result_acc); });
        } else {
          cgh.parallel_for(sycl::nd_range(m_r, m_r), [=](sycl::nd_item item) {
            test_action(acc_for_mptr, test_result_acc);
          });
        }
      });
    }
    // If expected value is equal to default value, then verification should be
    // skipped
    if (expected_results.op_return_result_val != m_default_expected_val) {
      CHECK(expected_results.op_return_result_val ==
            test_results.op_return_result_val);
    }
    // If expected value is equal to default value, then verification should be
    // skipped
    if (expected_results.op_calling_val != m_default_expected_val) {
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

    SECTION(section_name("Check multi_ptr operator++(multi_ptr& mp)")
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
      run_test(queue, run_test_action, verification_points);
    }
    SECTION(section_name("Check multi_ptr operator++(multi_ptr&, int)")
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
      run_test(queue, run_test_action, verification_points);
    }
    SECTION(section_name("Check multi_ptr operator--(multi_ptr&)")
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
      run_test(queue, run_test_action, verification_points);
    }
    SECTION(section_name("Check multi_ptr operator--(multi_ptr&, int)")
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
      run_test(queue, run_test_action, verification_points);
    }

    using diff_t = multi_ptr_t::difference_type;
    diff_t shift = m_array_size / 3;
    SECTION(section_name("Check multi_ptr operator+=(multi_ptr&, diff_type)")
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
      // Assing m_default_expected_val to skip verification of this value
      verification_points.op_return_result_val = m_default_expected_val;

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        multi_ptr_t mptr(acc_for_mptr);
        mptr += shift;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_calling_val = *mptr;
      };
      run_test(queue, run_test_action, verification_points);
    }
    SECTION(section_name("Check multi_ptr operator-=(multi_ptr&, diff_type)")
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
      // Assing m_default_expected_val to skip verification of this value
      verification_points.op_return_result_val = m_default_expected_val;

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
      run_test(queue, run_test_action, verification_points);
    }
    SECTION(
        section_name("Check multi_ptr operator+(const multi_ptr&, diff_type)")
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
      // Assing m_default_expected_val to skip verification of this value
      verification_points.op_calling_val = m_default_expected_val;

      const auto run_test_action = [=](auto acc_for_mptr,
                                       auto test_result_acc) {
        const multi_ptr_t mptr(acc_for_mptr);
        multi_ptr_t result_mptr = mptr + shift;

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *result_mptr;
      };
      run_test(queue, run_test_action, verification_points);
    }
    SECTION(
        section_name("Check multi_ptr operator-(const multi_ptr&, diff_type)")
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
      // Assing m_default_expected_val to skip verification of this value
      verification_points.op_calling_val = m_default_expected_val;

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
      run_test(queue, run_test_action, verification_points);
    }
    SECTION(section_name("Check multi_ptr operator*(const multi_ptr&)")
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
      // Assing m_default_expected_val to skip verification of this value
      verification_points.op_calling_val = m_default_expected_val;

      const auto run_test_action = [](auto acc_for_mptr, auto test_result_acc) {
        const multi_ptr_t mptr(acc_for_mptr);

        detail::test_results<T> &test_result = test_result_acc[0];
        test_result.op_return_result_val = *mptr;
      };
      run_test(queue, run_test_action, verification_points);
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
