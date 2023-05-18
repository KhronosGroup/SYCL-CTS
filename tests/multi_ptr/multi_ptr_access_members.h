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

//  Provides code for multi_ptr access members

#ifndef __SYCLCTS_TESTS_MULTI_PTR_ACCESS_MEMBERS_OPS_H
#define __SYCLCTS_TESTS_MULTI_PTR_ACCESS_MEMBERS_OPS_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_access_members {

namespace detail {

/**
 * @brief Structure that combines variables used to validate test results
 * @tparam T Data type that will be used in test
 */
template <typename T>
struct test_result {
  // Variables that will be used to check that access members returns correct
  // data type
  bool dereference_return_type_is_correct = false;
  bool dereference_op_return_type_is_correct = false;
  bool get_return_type_is_correct = false;
  bool get_raw_return_type_is_correct = false;
  bool get_decorated_return_type_is_correct = false;

  // Value that will be used to initialze variables with random values to avoid
  // false positive test result
  int value_to_init = 49;
  // Variables that will be used to check that access members returns correct
  // value
  T dereference_ret_value =
      user_def_types::get_init_value_helper<T>(value_to_init);
  T dereference_op_ret_value =
      user_def_types::get_init_value_helper<T>(value_to_init);
  T get_member_ret_value =
      user_def_types::get_init_value_helper<T>(value_to_init);
  T get_raw_member_ret_value =
      user_def_types::get_init_value_helper<T>(value_to_init);
};

}  // namespace detail

template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class kernel_access_members;

/**
 * @brief Provides functions for verification multi_ptr access operators that
 *        they returns correct data
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_access_members_tests {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  using multi_ptr_t = const sycl::multi_ptr<T, space, decorated>;

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
    auto queue = once_per_unit::get_queue();
    constexpr int val_to_init = 42;
    T value = user_def_types::get_init_value_helper<T>(val_to_init);

    // Variable that contains all variables that will be used to verify test
    // result
    detail::test_result<T> test_result;

    sycl::range r = sycl::range(1);
    {
      sycl::buffer<detail::test_result<T>> test_result_buffer(&test_result,
                                                              sycl::range(1));
      sycl::buffer<T> val_buffer(&value, sycl::range(1));
      queue.submit([&](sycl::handler &cgh) {
        auto test_result_acc =
            test_result_buffer.template get_access<sycl::access_mode::write>(
                cgh);
        auto test_device_code = [=](auto acc_for_multi_ptr) {
          const multi_ptr_t multi_ptr(acc_for_multi_ptr);
          detail::test_result<T> &test_result = test_result_acc[0];

          // Dereference and multi_ptr::operator->() available only when:
          // !std::is_void<sycl::multi_ptr::value_type>::value
          if constexpr (!std::is_void_v<typename multi_ptr_t::value_type>) {
            // Check dereference operator return value and type correctness
            test_result.dereference_return_type_is_correct =
                std::is_same_v<decltype(*multi_ptr),
                               typename multi_ptr_t::reference>;
            test_result.dereference_ret_value = *multi_ptr;
            // Check operator->() return value and type correctness
            test_result.dereference_op_return_type_is_correct =
                std::is_same_v<decltype(multi_ptr.operator->()),
                               typename multi_ptr_t::pointer>;
            test_result.dereference_op_ret_value = *(multi_ptr.operator->());
          }

          // Check get() return value and type correctness
          test_result.get_return_type_is_correct =
              std::is_same_v<decltype(multi_ptr.get()),
                             typename multi_ptr_t::pointer>;
          // Skip verification if pointer is decorated
          if constexpr (decorated == sycl::access::decorated::yes) {
            test_result.get_member_ret_value = *(multi_ptr.get());
          }
          // Check get_raw() return value and type correctness
          test_result.get_raw_return_type_is_correct = std::is_same_v<
              decltype(multi_ptr.get_raw()),
              std::add_pointer_t<typename multi_ptr_t::value_type>>;
          test_result.get_raw_member_ret_value = *(multi_ptr.get_raw());
          // Check get_decorated() return type correctness
          test_result.get_decorated_return_type_is_correct =
              std::is_pointer_v<decltype(multi_ptr.get_decorated())>;
        };

        using kname = kernel_access_members<T, AddrSpaceT, IsDecoratedT>;
        if constexpr (space == sycl::access::address_space::local_space) {
          sycl::local_accessor<T> acc_for_multi_ptr{sycl::range(1), cgh};
          cgh.parallel_for<kname>(
              sycl::nd_range(r, r), [=](sycl::nd_item<1> item) {
                value_operations::assign(acc_for_multi_ptr, value);
                sycl::group_barrier(item.get_group());
                test_device_code(acc_for_multi_ptr);
              });
        } else if constexpr (space ==
                             sycl::access::address_space::private_space) {
          cgh.single_task<kname>([=] {
            T priv_val = value;
            sycl::multi_ptr<T, sycl::access::address_space::private_space,
                            decorated>
                priv_val_mptr = sycl::address_space_cast<
                    sycl::access::address_space::private_space, decorated>(
                    &priv_val);
            test_device_code(priv_val_mptr);
          });
        } else {
          auto acc_for_multi_ptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task<kname>([=] { test_device_code(acc_for_multi_ptr); });
        }
      });
    }
    T expected_value = user_def_types::get_init_value_helper<T>(val_to_init);
    // Dereference and multi_ptr::operator->() available only when:
    // !std::is_void<sycl::multi_ptr::value_type>::value
    if constexpr (!std::is_void_v<typename multi_ptr_t::value_type>) {
      SECTION(sycl_cts::section_name("Check dereference return value and type")
                  .with("T", type_name)
                  .with("address_space", address_space_name)
                  .with("decorated", is_decorated_name)
                  .create()) {
        CHECK(test_result.dereference_return_type_is_correct);
        CHECK(test_result.dereference_ret_value == expected_value);
      }
      SECTION(
          sycl_cts::section_name(
              "Check dereference operator (operator->()) return value and type")
              .with("T", type_name)
              .with("address_space", address_space_name)
              .with("decorated", is_decorated_name)
              .create()) {
        CHECK(test_result.dereference_op_return_type_is_correct);
        CHECK(test_result.dereference_op_ret_value == expected_value);
      }
    }
    SECTION(sycl_cts::section_name("Check get() return value and type")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(test_result.get_return_type_is_correct);
      // Skip verification if pointer is decorated
      if constexpr (decorated == sycl::access::decorated::yes) {
        CHECK(test_result.get_member_ret_value == expected_value);
      }
    }
    SECTION(sycl_cts::section_name("Check get_raw() return value and type")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(test_result.get_raw_return_type_is_correct);
      CHECK(test_result.get_raw_member_ret_value == expected_value);
    }
    SECTION(sycl_cts::section_name("Check that get_decorated() returns pointer")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(test_result.get_decorated_return_type_is_correct);
    }
  }
};

template <typename T>
class check_multi_ptr_access_members_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();

    for_all_combinations<run_access_members_tests, T>(address_spaces,
                                                      is_decorated, type_name);
  }
};

}  // namespace multi_ptr_access_members

#endif  // __SYCLCTS_TESTS_MULTI_PTR_ACCESS_MEMBERS_OPS_H
