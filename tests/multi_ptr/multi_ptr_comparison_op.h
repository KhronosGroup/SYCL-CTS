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

//  Provides code for multi_ptr comparison operators

#ifndef __SYCLCTS_TESTS_MULTI_PTR_COMPARISON_OP_H
#define __SYCLCTS_TESTS_MULTI_PTR_COMPARISON_OP_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

#include <array>       // for std::array
#include <cstddef>     // for std::size_t
#include <functional>  // for std::less/greater/less_equal/greater_equal
#include <optional>    // for std::optional

namespace multi_ptr_comparison_op {

namespace detail {

/**
 * @brief Structure that combines variables used to validate test results for
 *        operators that compares two multi_ptr's
 * @tparam T Data type that will be used in test
 */
template <typename T>
struct mptr_mptr_test_results {
  // Use std::optional to disable verification if some member hasn't been
  // initialised by some value

  // Expected value that will be returned after "first_mptr *current operator*
  // first_mptr" operator will be called
  std::optional<bool> first_first_mptr_result_val;
  // Expected value that will be returned after "second_mptr *current operator*
  // first_mptr" operator will be called
  std::optional<bool> second_first_mptr_result_val;
  // Expected value that will be returned after "first_mptr *current operator*
  // second_mptr" operator will be called
  std::optional<bool> first_second_mptr_result_val;

  /**
   * @brief Verified test result. The object's data, that will call this
   *        function, will used as expected results
   * @param retrieved_results Retrieved test results
   */
  void verify_results(
      const mptr_mptr_test_results<T> &retrieved_results) const {
    // If expected value isn't initialized, then this verification should be
    // skipped
    if (first_first_mptr_result_val) {
      CHECK(first_first_mptr_result_val ==
            retrieved_results.first_first_mptr_result_val);
    }
    // If expected value isn't initialized, then this verification should be
    // skipped
    if (second_first_mptr_result_val) {
      CHECK(second_first_mptr_result_val ==
            retrieved_results.second_first_mptr_result_val);
    }
    // If expected value isn't initialized, then this verification should be
    // skipped
    if (first_second_mptr_result_val) {
      CHECK(first_second_mptr_result_val ==
            retrieved_results.first_second_mptr_result_val);
    }
  }
};

/**
 * @brief Structure that combines variables used to validate test results for
 *        operators that compares multi_ptr and nullptr
 * @tparam T Data type that will be used in test
 */
template <typename T>
struct mptr_nullptr_mptr_test_results {
  // Expected value that will be returned after "nullptr *current operator*
  // nullptr_mptr" operator will be called
  bool nullptr_nullptr_mptr_result_val = true;
  // Expected value that will be returned after "nullptr_mptr *current operator*
  // nullptr" operator will be called
  bool nullptr_mptr_nullptr_result_val = true;
  // Expected value that will be returned after "nullptr *current operator*
  // value_mptr" operator will be called
  bool nullptr_value_mptr_result_val = true;
  // Expected value that will be returned after "value_mptr *current operator*
  // nullptr" operator will be called
  bool value_mptr_nullptr_result_val = true;

  /**
   * @brief Verified test result. The object's data, that will call this
   *        function, will used as expected results
   * @param retrieved_results Retrieved test results
   */
  void verify_results(
      const mptr_nullptr_mptr_test_results<T> &retrieved_results) const {
    CHECK(nullptr_nullptr_mptr_result_val ==
          retrieved_results.nullptr_nullptr_mptr_result_val);
    CHECK(nullptr_mptr_nullptr_result_val ==
          retrieved_results.nullptr_mptr_nullptr_result_val);
    CHECK(nullptr_value_mptr_result_val ==
          retrieved_results.nullptr_value_mptr_result_val);
    CHECK(value_mptr_nullptr_result_val ==
          retrieved_results.value_mptr_nullptr_result_val);
  }
};

}  // namespace detail

template <typename T, typename AddrSpaceT, typename IsDecoratedT,
          typename KernelName>
class kernel_comparison_op;

/**
 * @brief Provides functor for verification on multi_ptr comparison operators
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_multi_ptr_comparison_op_test {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;

  using multi_ptr_t = sycl::multi_ptr<T, space, decorated>;

  const T m_small_value = user_def_types::get_init_value_helper<T>(1);
  const T m_great_value = user_def_types::get_init_value_helper<T>(2);
  // Use an array to be sure that we have two elements that has consecutive
  // memory addresses
  const T m_values_arr[2] = {m_small_value, m_great_value};
  static constexpr size_t m_values_arr_size =
      sizeof(m_values_arr) / sizeof(*m_values_arr);

  template <typename KernelName, typename TestActionT,
            typename ExpectedTestResultT>
  void run_test(sycl::queue &queue, TestActionT test_action,
                const ExpectedTestResultT &expected_results) {
    // Variable that contains all variables that will be used to verify test
    // result
    ExpectedTestResultT test_results;
    {
      sycl::range m_r(1);
      sycl::buffer<T> array_buffer(m_values_arr, m_values_arr_size);
      sycl::buffer<ExpectedTestResultT> test_result_buffer(&test_results, m_r);
      queue.submit([&](sycl::handler &cgh) {
        using kname =
            kernel_comparison_op<T, AddrSpaceT, IsDecoratedT, KernelName>;
        auto array_acc =
            array_buffer.template get_access<sycl::access_mode::read>(cgh);
        auto test_result_acc =
            test_result_buffer.template get_access<sycl::access_mode::write>(
                cgh);

        if constexpr (space == sycl::access::address_space::local_space) {
          sycl::local_accessor<T, 1> acc_for_mptr{
              sycl::range(m_values_arr_size), cgh};
          cgh.parallel_for<kname>(
              sycl::nd_range(m_r, m_r), [=](sycl::nd_item<1> item) {
                for (size_t i = 0; i < m_values_arr_size; ++i)
                  value_operations::assign(acc_for_mptr[i], array_acc[i]);
                sycl::group_barrier(item.get_group());
                test_action(acc_for_mptr, test_result_acc);
              });
        } else if constexpr (space ==
                             sycl::access::address_space::private_space) {
          cgh.single_task<kname>([=] {
            std::array<T, m_values_arr_size> priv_arr(
                multi_ptr_common::init_array<T, m_values_arr_size>::value);
            for (size_t i = 0; i < m_values_arr_size; ++i)
              value_operations::assign(priv_arr[i], array_acc[i]);
            sycl::multi_ptr<T, sycl::access::address_space::private_space,
                            decorated>
                priv_arr_mptr = sycl::address_space_cast<
                    sycl::access::address_space::private_space, decorated>(
                    priv_arr.data());
            test_action(priv_arr_mptr, test_result_acc);
          });
        } else {
          cgh.single_task<kname>(
              [=] { test_action(array_acc, test_result_acc); });
        }
      });
    }
    expected_results.verify_results(test_results);
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
    SECTION(
        sycl_cts::section_name(
            "Check multi_ptr operator==(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "first_mptr == first_mptr"
      // operator will be called
      expected_results.first_first_mptr_result_val = true;
      // Expected value that will be returned after "first_mptr == second_mptr"
      // operator will be called
      expected_results.first_second_mptr_result_val = false;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t arr_mptr(arr_acc);
        // multi_ptr that pointed at the first element
        multi_ptr_t mptr_1(arr_mptr);
        // multi_ptr that pointed at the second element
        multi_ptr_t mptr_2(arr_mptr + 1);

        auto &test_result = result_acc[0];

        test_result.first_first_mptr_result_val = mptr_1 == mptr_1;
        test_result.first_second_mptr_result_val = mptr_1 == mptr_2;
      };

      run_test<class eq>(queue, run_test_action, expected_results);
    }
    SECTION(
        sycl_cts::section_name(
            "Check multi_ptr operator!=(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "first_mptr != first_mptr"
      // operator will be called
      expected_results.first_first_mptr_result_val = false;
      // Expected value that will be returned after "first_mptr != second_mptr"
      // operator will be called
      expected_results.first_second_mptr_result_val = true;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t arr_mptr(arr_acc);
        // multi_ptr that pointed at the first element
        multi_ptr_t mptr_1(arr_mptr);
        // multi_ptr that pointed at the second element
        multi_ptr_t mptr_2(arr_mptr + 1);

        auto &test_result = result_acc[0];

        test_result.first_first_mptr_result_val = mptr_1 != mptr_1;
        test_result.first_second_mptr_result_val = mptr_1 != mptr_2;
      };

      run_test<class neq>(queue, run_test_action, expected_results);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator<(const multi_ptr&, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "first_mptr < first_mptr"
      // operator will be called
      expected_results.first_first_mptr_result_val = false;
      // Expected value that will be returned after "first_mptr < second_mptr"
      // operator will be called
      expected_results.first_second_mptr_result_val = true;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t arr_mptr(arr_acc);
        // multi_ptr that pointed at the first element
        multi_ptr_t mptr_1(arr_mptr);
        // multi_ptr that pointed at the second element
        multi_ptr_t mptr_2(arr_mptr + 1);

        auto &test_result = result_acc[0];

        test_result.first_first_mptr_result_val = mptr_1 < mptr_1;
        test_result.first_second_mptr_result_val = mptr_1 < mptr_2;
      };

      run_test<class lt>(queue, run_test_action, expected_results);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator>(const multi_ptr&, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "first_mptr > second_mptr"
      // operator will be called
      expected_results.first_second_mptr_result_val = false;
      // Expected value that will be returned after "second_mptr > first_mptr"
      // operator will be called
      expected_results.second_first_mptr_result_val = true;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t arr_mptr(arr_acc);
        // multi_ptr that pointed at the first element
        multi_ptr_t mptr_1(arr_mptr);
        // multi_ptr that pointed at the second element
        multi_ptr_t mptr_2(arr_mptr + 1);

        auto &test_result = result_acc[0];

        test_result.first_second_mptr_result_val = mptr_1 > mptr_2;
        test_result.second_first_mptr_result_val = mptr_2 > mptr_1;
      };

      run_test<class gt>(queue, run_test_action, expected_results);
    }
    SECTION(
        sycl_cts::section_name(
            "Check multi_ptr operator<=(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "first_mptr <= first_mptr"
      // operator will be called
      expected_results.first_first_mptr_result_val = true;
      // Expected value that will be returned after "first_mptr <= second_mptr"
      // operator will be called
      expected_results.first_second_mptr_result_val = true;
      // Expected value that will be returned after "second_mptr <= first_mptr"
      // operator will be called
      expected_results.second_first_mptr_result_val = false;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t arr_mptr(arr_acc);
        // multi_ptr that pointed at the first element
        multi_ptr_t mptr_1(arr_mptr);
        // multi_ptr that pointed at the second element
        multi_ptr_t mptr_2(arr_mptr + 1);

        auto &test_result = result_acc[0];

        test_result.first_first_mptr_result_val = mptr_1 <= mptr_1;
        test_result.first_second_mptr_result_val = mptr_1 <= mptr_2;
        test_result.second_first_mptr_result_val = mptr_2 <= mptr_1;
      };

      run_test<class leq>(queue, run_test_action, expected_results);
    }
    SECTION(
        sycl_cts::section_name(
            "Check multi_ptr operator>=(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "first_mptr >= first_mptr"
      // operator will be called
      expected_results.first_first_mptr_result_val = true;
      // Expected value that will be returned after "second_mptr >= first_mptr"
      // operator will be called
      expected_results.first_second_mptr_result_val = true;
      // Expected value that will be returned after "first_mptr >= second_mptr"
      // operator will be called
      expected_results.second_first_mptr_result_val = false;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t arr_mptr(arr_acc);
        // multi_ptr that pointed at the first element
        multi_ptr_t mptr_1(arr_mptr);
        // multi_ptr that pointed at the second element
        multi_ptr_t mptr_2(arr_mptr + 1);

        auto &test_result = result_acc[0];

        test_result.first_first_mptr_result_val = mptr_1 >= mptr_1;
        test_result.first_second_mptr_result_val = mptr_2 >= mptr_1;
        test_result.second_first_mptr_result_val = mptr_1 >= mptr_2;
      };

      run_test<class geq>(queue, run_test_action, expected_results);
    }
    SECTION(
        sycl_cts::section_name(
            "Check multi_ptr operator==(const multi_ptr& lhs, std::nullptr_t)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_nullptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "nullptr_mptr == nullptr"
      // operator will be called
      expected_results.nullptr_nullptr_mptr_result_val = true;
      // Expected value that will be returned after "nullptr == nullptr_mptr"
      // operator will be called
      expected_results.nullptr_mptr_nullptr_result_val = true;
      // Expected value that will be returned after "nullptr == value_mptr"
      // operator will be called
      expected_results.nullptr_value_mptr_result_val = false;
      // Expected value that will be returned after "value_mptr == nullptr"
      // operator will be called
      expected_results.value_mptr_nullptr_result_val = false;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(arr_acc);

        auto &test_result = result_acc[0];

        test_result.nullptr_nullptr_mptr_result_val = nullptr_mptr == nullptr;
        test_result.nullptr_mptr_nullptr_result_val = nullptr == nullptr_mptr;
        test_result.nullptr_value_mptr_result_val = value_mptr == nullptr;
        test_result.value_mptr_nullptr_result_val = nullptr == value_mptr;
      };

      run_test<class eq_null>(queue, run_test_action, expected_results);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator!=(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result
      detail::mptr_nullptr_mptr_test_results<T> expected_results;
      // Expected value that will be returned after "nullptr != nullptr_mptr"
      // operator will be called
      expected_results.nullptr_nullptr_mptr_result_val = false;
      // Expected value that will be returned after "nullptr_mptr != nullptr"
      // operator will be called
      expected_results.nullptr_mptr_nullptr_result_val = false;
      // Expected value that will be returned after "nullptr != value_mptr"
      // operator will be called
      expected_results.nullptr_value_mptr_result_val = true;
      // Expected value that will be returned after "value_mptr != nullptr"
      // operator will be called
      expected_results.value_mptr_nullptr_result_val = true;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(arr_acc);

        auto &test_result = result_acc[0];

        test_result.nullptr_nullptr_mptr_result_val = nullptr != nullptr_mptr;
        test_result.nullptr_mptr_nullptr_result_val = nullptr_mptr != nullptr;
        test_result.nullptr_value_mptr_result_val = nullptr != value_mptr;
        test_result.value_mptr_nullptr_result_val = value_mptr != nullptr;
      };

      run_test<class neq_null>(queue, run_test_action, expected_results);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator<(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result. We expect that all of them will be true.
      detail::mptr_nullptr_mptr_test_results<T> expected_results;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(arr_acc);

        auto &test_result = result_acc[0];

        test_result.nullptr_nullptr_mptr_result_val =
            std::less<multi_ptr_t>()(nullptr, nullptr_mptr) ==
            nullptr < nullptr_mptr;
        test_result.nullptr_mptr_nullptr_result_val =
            std::less<multi_ptr_t>()(nullptr_mptr, nullptr) ==
            nullptr_mptr < nullptr;
        test_result.nullptr_value_mptr_result_val =
            std::less<multi_ptr_t>()(nullptr, value_mptr) ==
            nullptr < value_mptr;
        test_result.value_mptr_nullptr_result_val =
            std::less<multi_ptr_t>()(value_mptr, nullptr) ==
            value_mptr < nullptr;
      };

      run_test<class lt_null>(queue, run_test_action, expected_results);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator>(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result. We expect that all of them will be true.
      detail::mptr_nullptr_mptr_test_results<T> expected_results;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(arr_acc);

        auto &test_result = result_acc[0];

        test_result.nullptr_nullptr_mptr_result_val =
            std::greater<multi_ptr_t>()(nullptr, nullptr_mptr) ==
            nullptr > nullptr_mptr;
        test_result.nullptr_mptr_nullptr_result_val =
            std::greater<multi_ptr_t>()(nullptr_mptr, nullptr) ==
            nullptr_mptr > nullptr;
        test_result.nullptr_value_mptr_result_val =
            std::greater<multi_ptr_t>()(nullptr, value_mptr) ==
            nullptr > value_mptr;
        test_result.value_mptr_nullptr_result_val =
            std::greater<multi_ptr_t>()(value_mptr, nullptr) ==
            value_mptr > nullptr;
      };

      run_test<class gt_null>(queue, run_test_action, expected_results);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator<=(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result. We expect that all of them will be true.
      detail::mptr_nullptr_mptr_test_results<T> expected_results;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(arr_acc);

        auto &test_result = result_acc[0];

        test_result.nullptr_nullptr_mptr_result_val =
            std::less_equal<multi_ptr_t>()(nullptr, nullptr_mptr) ==
            nullptr <= nullptr_mptr;
        test_result.nullptr_mptr_nullptr_result_val =
            std::less_equal<multi_ptr_t>()(nullptr_mptr, nullptr) ==
            nullptr_mptr <= nullptr;
        test_result.nullptr_value_mptr_result_val =
            std::less_equal<multi_ptr_t>()(nullptr, value_mptr) ==
            nullptr <= value_mptr;
        test_result.value_mptr_nullptr_result_val =
            std::less_equal<multi_ptr_t>()(value_mptr, nullptr) ==
            value_mptr <= nullptr;
      };

      run_test<class leq_null>(queue, run_test_action, expected_results);
    }
    SECTION(sycl_cts::section_name(
                "Check multi_ptr operator>=(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Variable that contains all variables that will be used to verify test
      // result. We expect that all of them will be true.
      detail::mptr_nullptr_mptr_test_results<T> expected_results;

      const auto run_test_action = [](auto arr_acc, auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(arr_acc);

        auto &test_result = result_acc[0];

        test_result.nullptr_nullptr_mptr_result_val =
            std::greater_equal<multi_ptr_t>()(nullptr, nullptr_mptr) ==
            nullptr >= nullptr_mptr;
        test_result.nullptr_mptr_nullptr_result_val =
            std::greater_equal<multi_ptr_t>()(nullptr_mptr, nullptr) ==
            nullptr_mptr >= nullptr;
        test_result.nullptr_value_mptr_result_val =
            std::greater_equal<multi_ptr_t>()(nullptr, value_mptr) ==
            nullptr >= value_mptr;
        test_result.value_mptr_nullptr_result_val =
            std::greater_equal<multi_ptr_t>()(value_mptr, nullptr) ==
            value_mptr >= nullptr;
      };

      run_test<class geq_null>(queue, run_test_action, expected_results);
    }
  }
};

template <typename T>
class check_multi_ptr_comparison_op_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();
    // Run test
    for_all_combinations<run_multi_ptr_comparison_op_test, T>(
        address_spaces, is_decorated, type_name);
  }
};

}  // namespace multi_ptr_comparison_op

#endif  // __SYCLCTS_TESTS_MULTI_PTR_COMPARISON_OP_H
