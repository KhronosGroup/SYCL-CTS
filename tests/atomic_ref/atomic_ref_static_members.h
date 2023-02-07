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
//  Provides tests for sycl::atomic_ref static members
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_STATIC_MEMBERS_H
#define SYCL_CTS_ATOMIC_REF_STATIC_MEMBERS_H

#include "atomic_ref_common.h"

namespace atomic_ref::static_members {
using namespace sycl_cts;
using namespace atomic_ref::tests::common;

template <typename T, typename OrderT, typename ScopeT, typename AS>
class run_static_member_tests {
  static constexpr sycl::memory_order MemoryOrder = OrderT::value;
  static constexpr sycl::memory_scope MemoryScope = ScopeT::value;
  static constexpr sycl::access::address_space AddressSpace = AS::value;

  using atomic_ref_type =
      sycl::atomic_ref<T, MemoryOrder, MemoryScope, AddressSpace>;

 public:
  void operator()(const std::string& type_name,
                  const std::string& memory_order_name,
                  const std::string& memory_scope_name,
                  const std::string& address_space_name) {
    STATIC_CHECK(std::is_same_v<T, typename atomic_ref_type::value_type>);

    if constexpr (!std::is_pointer_v<T>) {
      STATIC_CHECK(std::is_same_v<typename atomic_ref_type::difference_type,
                                  typename atomic_ref_type::value_type>);
    } else {
      STATIC_CHECK(
          std::is_same_v<typename atomic_ref_type::difference_type, ptrdiff_t>);
    }

    STATIC_CHECK(std::is_same_v<decltype(atomic_ref_type::required_alignment),
                                const size_t>);
    STATIC_CHECK(atomic_ref_type::required_alignment >=
                 alignof(std::remove_pointer_t<T>));

    STATIC_CHECK(std::is_same_v<decltype(atomic_ref_type::is_always_lock_free),
                                const bool>);

    STATIC_CHECK(atomic_ref_type::default_read_order ==
                 (MemoryOrder == sycl::memory_order::acq_rel
                      ? sycl::memory_order::acquire
                      : MemoryOrder));
    STATIC_CHECK(
        std::is_const_v<decltype(atomic_ref_type::default_read_order)>);

    STATIC_CHECK(atomic_ref_type::default_write_order ==
                 (MemoryOrder == sycl::memory_order::acq_rel
                      ? sycl::memory_order::release
                      : MemoryOrder));
    STATIC_CHECK(
        std::is_const_v<decltype(atomic_ref_type::default_write_order)>);

    STATIC_CHECK(atomic_ref_type::default_read_modify_write_order ==
                 MemoryOrder);
    STATIC_CHECK(std::is_const_v<
                 decltype(atomic_ref_type::default_read_modify_write_order)>);

    STATIC_CHECK(atomic_ref_type::default_scope == MemoryScope);
    STATIC_CHECK(std::is_const_v<decltype(atomic_ref_type::default_scope)>);
  }
};

/**
 * @brief Run tests for sycl::atomic_ref static members
 */
template <typename T>
class run_test {
 public:
  void operator()(const std::string& type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<run_static_member_tests, T>(
        memory_orders, memory_scopes, address_spaces, type_name);

    std::string type_name_for_pointer_types = type_name + "*";
    for_all_combinations<run_static_member_tests, T*>(
        memory_orders, memory_scopes, address_spaces,
        type_name_for_pointer_types);
  }
};

}  // namespace atomic_ref::static_members

#endif  // SYCL_CTS_ATOMIC_REF_STATIC_MEMBERS_H
