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
//  Provides tests for SYCL atomic_ref api
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_API_TESTS_H
#define SYCL_CTS_ATOMIC_REF_API_TESTS_H

#include "atomic_ref_T_op_test.h"
#include "atomic_ref_add_sub_op_all_types_test.h"
#include "atomic_ref_assign_op_test.h"
#include "atomic_ref_bitwise_op_test.h"
#include "atomic_ref_common.h"
#include "atomic_ref_compare_exchange_test.h"
#include "atomic_ref_exchange_test.h"
#include "atomic_ref_fetch_add_sub_all_types_test.h"
#include "atomic_ref_fetch_bitwise_test.h"
#include "atomic_ref_fetch_min_max_test.h"
#include "atomic_ref_incr_decr_op_test.h"
#include "atomic_ref_is_lock_free_test.h"
#include "atomic_ref_store_test.h"
#include <limits>
#include <memory>
#include <type_traits>

namespace atomic_ref::tests::api {
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class run_api_tests;

template <typename T>
struct run_tests {
  void operator()(const std::string& type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<run_api_tests, T>(memory_orders, memory_scopes,
                                           address_spaces, type_name);

    std::string type_name_for_pointer_types = type_name + "*";
    for_all_combinations<run_api_tests, T*>(memory_orders, memory_scopes,
                                            address_spaces,
                                            type_name_for_pointer_types);
  }
};

template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class run_api_tests {
  static constexpr sycl::memory_order memory_orders[] = {
      sycl::memory_order::relaxed, sycl::memory_order::acq_rel,
      sycl::memory_order::seq_cst};
  static constexpr sycl::memory_scope memory_scopes[] = {
      sycl::memory_scope::work_item, sycl::memory_scope::sub_group,
      sycl::memory_scope::work_group, sycl::memory_scope::device,
      sycl::memory_scope::system};

  using atomic_ref_test =
      atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT>;

  sycl::queue queue;
  std::vector<typename atomic_ref_test::sptr> tests;

 public:
  run_api_tests() : queue(util::get_cts_object::queue()) {
    if (memory_order_and_scope_are_supported(queue, MemoryOrderT::value,
                                             MemoryScopeT::value)) {
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_is_lock_free_test<T, MemoryOrderT, MemoryScopeT,
                                           AddressSpaceT>(queue)));
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_store_test<T, MemoryOrderT, MemoryScopeT,
                                    AddressSpaceT>(queue)));
// FIXME: re-enable when atomic_ref<T*>::operator=() is implemented
#if !SYCL_CTS_COMPILING_WITH_DPCPP
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_assign_op_test<T, MemoryOrderT, MemoryScopeT,
                                        AddressSpaceT>(queue)));
#endif
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_T_op_test<T, MemoryOrderT, MemoryScopeT,
                                   AddressSpaceT>(queue)));
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_exchange_test<T, MemoryOrderT, MemoryScopeT,
                                       AddressSpaceT>(queue)));
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_compare_exchange_test<T, MemoryOrderT, MemoryScopeT,
                                               AddressSpaceT, weak>(queue)));
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_compare_exchange_test<T, MemoryOrderT, MemoryScopeT,
                                               AddressSpaceT, strong>(queue)));
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_fetch_add_sub_all_types_test<
              T, MemoryOrderT, MemoryScopeT, AddressSpaceT>(queue)));
      tests.push_back(typename atomic_ref_test::sptr(
          new atomic_ref_add_sub_op_all_types_test<
              T, MemoryOrderT, MemoryScopeT, AddressSpaceT>(queue)));
      if constexpr (std::is_integral_v<T>) {
        tests.push_back(typename atomic_ref_test::sptr(
            new atomic_ref_fetch_bitwise_test<T, MemoryOrderT, MemoryScopeT,
                                              AddressSpaceT>(queue)));
        tests.push_back(typename atomic_ref_test::sptr(
            new atomic_ref_bitwise_op_test<T, MemoryOrderT, MemoryScopeT,
                                           AddressSpaceT>(queue)));
      }
      if constexpr (std::is_integral_v<T> or std::is_pointer_v<T>) {
        tests.push_back(typename atomic_ref_test::sptr(
            new atomic_ref_incr_decr_op_test<T, MemoryOrderT, MemoryScopeT,
                                             AddressSpaceT>(queue)));
      }
      if constexpr (std::is_integral_v<T> or std::is_floating_point_v<T>) {
        tests.push_back(typename atomic_ref_test::sptr(
            new atomic_ref_fetch_min_max_test<T, MemoryOrderT, MemoryScopeT,
                                              AddressSpaceT>(queue)));
      }
    }
  }

  void operator()(const std::string& type_name, const std::string& memory_order,
                  const std::string& memory_scope,
                  const std::string& address_space) {
    for (typename atomic_ref_test::sptr test : tests) {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
      if (test->require_combination_for_full_conformance()) {
        for (auto order : memory_orders) {
          for (auto scope : memory_scopes) {
            test->run_test(type_name, memory_order, memory_scope, address_space,
                           order, scope);
          }
        }
      } else {
        test->run_test(type_name, memory_order, memory_scope, address_space);
      }
#else
      test->run_test(type_name, memory_order, memory_scope, address_space);
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
    }
  }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_API_TESTS_H
