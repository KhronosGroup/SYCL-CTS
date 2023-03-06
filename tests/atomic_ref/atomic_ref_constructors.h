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
//  Provides tests for sycl::atomic_ref constructors.
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_CONSTRUCTORS_H
#define SYCL_CTS_ATOMIC_REF_CONSTRUCTORS_H

#include "atomic_ref_common.h"
#include <type_traits>

namespace atomic_ref::tests::constructors {
using namespace atomic_ref::tests::common;

template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class run_constructor_tests {
  static constexpr sycl::memory_order MemoryOrder = MemoryOrderT::value;
  static constexpr sycl::memory_scope MemoryScope = MemoryScopeT::value;
  static constexpr sycl::access::address_space AddressSpace =
      AddressSpaceT::value;
  using atomic_ref_type =
      sycl::atomic_ref<T, MemoryOrder, MemoryScope, AddressSpace>;
  using referenced_type = std::remove_pointer_t<T>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &memory_order_name,
                  const std::string &memory_scope_name,
                  const std::string &address_space_name) {
    auto queue = util::get_cts_object::queue();
    if (memory_order_and_scope_are_not_supported(queue, MemoryOrder,
                                                 MemoryScope)) {
      return;
    }
    referenced_type data =
        value_operations::init<referenced_type>(expected_val);
    T value;
    if constexpr (std::is_pointer_v<T>)
      value = &data;
    else
      value = data;

    SECTION(get_section_name(type_name, memory_order_name, memory_scope_name,
                             address_space_name,
                             "Check constructors on device")) {
      if constexpr (AddressSpace != sycl::access::address_space::global_space) {
        std::array res{false, false, false};
        {
          sycl::buffer result_buf(res.data(), sycl::range(2));

          queue
              .submit([&](sycl::handler &cgh) {
                auto result_accessor =
                    result_buf.template get_access<sycl::access_mode::write>(
                        cgh);
                sycl::local_accessor<T, 1> loc_acc(sycl::range<1>(1), cgh);
                cgh.parallel_for(sycl::nd_range<1>(1, 1),
                                 [=](sycl::nd_item<1>) {
                                   loc_acc[0] = value;
                                   atomic_ref_type a_r(loc_acc[0]);

                                   auto result = a_r.load();
                                   result_accessor[0] = (result == value);

                                   atomic_ref_type a_r_copy(a_r);
                                   auto result_copy = a_r.load();
                                   result_accessor[1] = (result_copy == value);

                                   result_accessor[2] =
                                       std::is_same_v<decltype(result), T>;
                                 });
              })
              .wait_and_throw();
        }
        {
          INFO("check atomic_ref(T&) (local space)");
          CHECK(res[0]);
        }
        {
          INFO("check copy-constructor (local space)");
          CHECK(res[1]);
        }
        {
          INFO("Error returned type for load() (local space)");
          CHECK(res[2]);
        }
      }
      if constexpr (AddressSpace != sycl::access::address_space::local_space) {
        std::array res{false, false, false};
        {
          sycl::buffer data_buf(&value, sycl::range(1));
          sycl::buffer result_buf(res.data(), sycl::range(2));

          queue
              .submit([&](sycl::handler &cgh) {
                auto data_accessor =
                    data_buf.template get_access<sycl::access_mode::read_write>(
                        cgh);
                auto result_accessor =
                    result_buf.template get_access<sycl::access_mode::write>(
                        cgh);
                cgh.single_task([=] {
                  atomic_ref_type a_r(data_accessor[0]);

                  auto result = a_r.load();
                  result_accessor[0] = (result == value);

                  atomic_ref_type a_r_copy(a_r);
                  auto result_copy = a_r.load();
                  result_accessor[1] = (result_copy == value);

                  result_accessor[2] = std::is_same_v<decltype(result), T>;
                });
              })
              .wait_and_throw();
        }
        {
          INFO("check atomic_ref(T&) (global space)");
          CHECK(res[0]);
        }
        {
          INFO("check copy-constructor (global space)");
          CHECK(res[1]);
        }
        {
          INFO("Error returned type for load() (global space)");
          CHECK(res[2]);
        }
      }
    }
  }
};

/**
 * @brief Run tests for sycl::atomic_ref constructor
 */
template <typename T>
struct run_test {
  void operator()(const std::string &type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<run_constructor_tests, T>(memory_orders, memory_scopes,
                                                   address_spaces, type_name);

    if (is_64_bits_pointer<T *>() && device_has_not_aspect_atomic64()) return;

    std::string type_name_for_pointer_types = type_name + "*";
    for_all_combinations<run_constructor_tests, T *>(
        memory_orders, memory_scopes, address_spaces,
        type_name_for_pointer_types);
  }
};

}  // namespace atomic_ref::tests::constructors

#endif  // SYCL_CTS_ATOMIC_REF_CONSTRUCTORS_H
