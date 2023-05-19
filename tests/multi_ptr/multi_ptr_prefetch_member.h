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

//  Provides code for multi_ptr prefetch member

#ifndef __SYCLCTS_TESTS_MULTI_PTR_PREFETCH_FUNC_H
#define __SYCLCTS_TESTS_MULTI_PTR_PREFETCH_FUNC_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

#include <type_traits>  // for std::is_same

namespace multi_ptr_prefetch_member {

template <typename T, typename IsDecoratedT>
class kernel_prefetch_member;

constexpr int expected_val = 42;

/**
 * @brief Provides functions for verification multi_ptr prefetch member
 * @tparam T Current data type
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename IsDecoratedT>
class run_prefetch_test {
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  using multi_ptr_t =
      sycl::multi_ptr<T, sycl::access::address_space::global_space, decorated>;

 public:
  /**
   * @param type_name Current data type string representation
   * @param is_decorated_name Current sycl::access::decorated string
   *        representation
   */
  void operator()(const std::string &type_name,
                  const std::string &is_decorated_name) {
    static_assert(!std::is_same_v<T, void>,
                  "Data type shouldn't be is same to void type");

    auto queue = once_per_unit::get_queue();
    T value = user_def_types::get_init_value_helper<T>(expected_val);
    SECTION(sycl_cts::section_name("Check multi_ptr::prefetch()")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      bool res = false;
      {
        sycl::buffer<bool> res_buf(&res, sycl::range(1));
        sycl::buffer<T> val_buffer(&value, sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task<kernel_prefetch_member<T, IsDecoratedT>>([=] {
            const multi_ptr_t mptr(acc_for_mptr);

            // Check call and const correctness for multi_ptr::prefetch(), then
            // verify that multi_ptr contained expected value
            mptr.prefetch(0);
            // Check that data is not corrupted
            res_acc[0] = mptr[0] == acc_for_mptr[0];
          });
        });
      }
      // Check that data in multi_ptr is not corrupted
      CHECK(res);
    }
  }
};

template <typename T>
class check_multi_ptr_prefetch_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto is_decorated = multi_ptr_common::get_decorated();
    // Run test
    for_all_combinations<run_prefetch_test, T>(is_decorated, type_name);
  }
};

}  // namespace multi_ptr_prefetch_member

#endif  // __SYCLCTS_TESTS_MULTI_PTR_PREFETCH_FUNC_H
