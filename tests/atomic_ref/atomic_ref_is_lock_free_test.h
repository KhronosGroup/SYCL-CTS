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
//  Provides tests for sycl::atomic_ref is_lock_free() method
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_IS_LOCK_FREE_TEST_H
#define SYCL_CTS_ATOMIC_REF_IS_LOCK_FREE_TEST_H

#include "atomic_ref_test_base.h"

namespace atomic_ref::tests::api {
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomic_ref_is_lock_free_test
    : public atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT> {
  using base = atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT>;
  using base::atomic_ref_test;

 public:
  void run_on_device(const std::string& type_name,
                     const std::string& memory_order,
                     const std::string& memory_scope,
                     const std::string& address_space,
                     sycl::memory_order memory_order_val =
                         base::memory_order_for_atomic_ref_obj,
                     sycl::memory_scope memory_scope_val =
                         base::memory_scope_for_atomic_ref_obj) {
    std::string description =
        get_section_name(type_name, memory_order, memory_scope, address_space,
                         "Check is_lock_free() method");
    auto is_lock_free_test = [](T val_expd, T val_chgd,
                                typename base::AtomicRT& a_r, auto result_acc,
                                auto ref_data_acc) {
      auto lock = a_r.is_lock_free();
      if constexpr (base::AtomicRT::is_always_lock_free == true) {
        result_acc[0] = (lock == true);
      }
      result_acc[1] = std::is_same_v<decltype(lock), bool>;
    };
    if constexpr (base::address_space_is_not_local_space()) {
      std::array result{false, false};
      this->queue_submit_global_scope(result, is_lock_free_test);
      if constexpr (base::AtomicRT::is_always_lock_free == true) {
        INFO(description + " (global space)" + "\nError returned value");
        CHECK(result[0]);
      }
      INFO(description + " (global space)" + "\nError returned type");
      CHECK(result[1]);
    }

    if constexpr (base::address_space_is_not_global_space()) {
      std::array result{false, false};
      this->queue_submit_local_scope(result, is_lock_free_test);
      if constexpr (base::AtomicRT::is_always_lock_free == true) {
        INFO(description + " (local space)" + "\nError returned value");
        CHECK(result[0]);
      }
      INFO(description + " (local space)" + "\nError returned type");
      CHECK(result[1]);
    }
  }

  bool require_combination_for_full_conformance() { return false; }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_IS_LOCK_FREE_TEST_H
