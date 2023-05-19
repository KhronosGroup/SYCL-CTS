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
#ifndef __SYCLCTS_TESTS_MULTI_PTR_SUBSCRIPT_OP_H
#define __SYCLCTS_TESTS_MULTI_PTR_SUBSCRIPT_OP_H

#include "multi_ptr_common.h"

namespace multi_ptr_subscript_op {

/**
 * @brief Provides functions for verification multi_ptr subscript operator that
 *        it returns correct data
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_subscript_op_tests {
  static constexpr size_t array_size = 10;
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  using multi_ptr_t = const sycl::multi_ptr<T, space, decorated>;
  using arr_t = std::array<T, array_size>;

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
    bool ret_type_check;

    // arrays for returned and expected values
    arr_t ret_arr = multi_ptr_common::init_array<T, array_size>::value;
    arr_t ret_arr_neg = multi_ptr_common::init_array<T, array_size>::value;
    arr_t exp_arr = multi_ptr_common::init_array<T, array_size>::value;

    std::iota(exp_arr.begin(), exp_arr.end(), 0);

    {
      sycl::buffer<bool> ret_type_buffer(&ret_type_check, {1});
      sycl::buffer<T> ret_arr_buffer(ret_arr.data(), {array_size});
      sycl::buffer<T> ret_arr_neg_buffer(ret_arr_neg.data(), {array_size});
      sycl::buffer<T> exp_arr_buffer(exp_arr.data(), {array_size});

      queue.submit([&](sycl::handler &cgh) {
        auto ret_type_acc =
            ret_type_buffer.template get_access<sycl::access_mode::write>(cgh);
        auto ret_arr_acc =
            ret_arr_buffer.template get_access<sycl::access_mode::write>(cgh);
        auto ret_arr_neg_acc =
            ret_arr_neg_buffer.template get_access<sycl::access_mode::write>(
                cgh);

        auto test_device_code = [=](auto acc_for_multi_ptr,
                                    auto acc_for_multi_ptr_negative) {
          multi_ptr_t multi_ptr(acc_for_multi_ptr);
          multi_ptr_t multi_ptr_negative(acc_for_multi_ptr_negative);

          // standard indexation
          for (int i = 0; i < array_size; i++) {
            ret_arr_acc[i] = multi_ptr[i];
          }

          // negative indexation
          for (int i = 0; i < array_size; i++) {
            ret_arr_neg_acc[i] = multi_ptr_negative[-i];
          }

          // type check
          ret_type_acc[0] = std::is_same_v<decltype(multi_ptr[0]),
                                           typename multi_ptr_t::reference>;
        };

        if constexpr (space == sycl::access::address_space::local_space) {
          sycl::local_accessor<T> exp_arr_acc{{array_size}, cgh};
          sycl::local_accessor<T> exp_arr_neg_acc{{array_size}, cgh};
          cgh.single_task([=] {
            value_operations::assign(exp_arr_acc, exp_arr);
            value_operations::assign(exp_arr_neg_acc, exp_arr);

            // pointer to the end of the array
            T *arr_end = &exp_arr_neg_acc[array_size - 1];
            multi_ptr_t multi_ptr_negative =
                sycl::address_space_cast<space, decorated>(arr_end);

            test_device_code(exp_arr_acc, multi_ptr_negative);
          });
        } else if constexpr (space ==
                             sycl::access::address_space::private_space) {
          cgh.single_task([=] {
            T *priv_arr = const_cast<T *>(&(exp_arr[0]));
            multi_ptr_t priv_arr_mptr =
                sycl::address_space_cast<space, decorated>(priv_arr);
            multi_ptr_t multi_ptr_negative =
                sycl::address_space_cast<space, decorated>(priv_arr +
                                                           array_size - 1);
            test_device_code(priv_arr_mptr, multi_ptr_negative);
          });
        } else {
          auto exp_arr_acc =
              exp_arr_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto exp_arr_neg_acc =
              exp_arr_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task([=] {
            T *arr_end = const_cast<T *>(&exp_arr_neg_acc[array_size - 1]);
            multi_ptr_t multi_ptr_negative =
                sycl::address_space_cast<space, decorated>(arr_end);
            test_device_code(exp_arr_acc, multi_ptr_negative);
          });
        }
      });
    }

    std::reverse(ret_arr_neg.begin(), ret_arr_neg.end());

    // multi_ptr::operator[](std::ptrdiff_t) available only when:
    // !std::is_void<sycl::multi_ptr::value_type>::value
    if constexpr (!std::is_void_v<typename multi_ptr_t::value_type>) {
      INFO("Check operator[](std::ptrdiff_t) return value and type"
           << "T" << type_name << "address_space" << address_space_name
           << "decorated" << is_decorated_name);
      CHECK(ret_arr == exp_arr);
      CHECK(ret_arr_neg == exp_arr);
      CHECK(ret_type_check);
    }
  }
};

template <typename T>
class check_multi_ptr_subscript_op {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();
    for_all_combinations<run_subscript_op_tests, T>(address_spaces,
                                                    is_decorated, type_name);
  }
};

}  // namespace multi_ptr_subscript_op

#endif  // __SYCLCTS_TESTS_MULTI_PTR_SUBSCRIPT_OP_H
