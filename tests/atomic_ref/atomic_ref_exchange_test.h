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
//  Provides tests for sycl::atomic_ref exchange() method
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_EXCHANGE_TEST_H
#define SYCL_CTS_ATOMIC_REF_EXCHANGE_TEST_H

#include "atomic_ref_test_base.h"

namespace atomic_ref::tests::api {
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomic_ref_exchange_test
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
                         memory_order_val, memory_scope_val,
                         "Check if exchange() method replaces the value of "
                         "the object referenced by this atomic_ref with"
                         " value operand and returns the original value of "
                         "the referenced object in device code");
    auto exchange_test = [memory_order_val, memory_scope_val](
                             T val_expd, T val_chgd,
                             typename base::AtomicRT& a_r, auto result_acc,
                             auto ref_data_acc) {
      T original_val = val_expd;
      auto ref_val_before_exchange =
          a_r.exchange(val_chgd, memory_order_val, memory_scope_val);
      result_acc[0] = ref_val_before_exchange == original_val;
      result_acc[1] = ref_data_acc[0] == val_chgd;
      result_acc[2] = std::is_same_v<decltype(ref_val_before_exchange), T>;
    };

    if constexpr (base::address_space_is_not_local_space()) {
      std::array result{false, false, false};
      this->queue_submit_global_scope(result, exchange_test);
      {
        INFO(description + "\nCheck returned val (global space)");
        CHECK(result[0]);
      }
      {
        INFO(description +
             "\nCheck that referenced val is updated (global space)");
        CHECK(result[1]);
      }
      {
        INFO(description + "\nError returned type (global space)");
        CHECK(result[2]);
      }
    }

    if constexpr (base::address_space_is_not_global_space()) {
      std::array result{false, false, false};
      this->queue_submit_local_scope(result, exchange_test);
      {
        INFO(description + "\nCheck returned val (local space)");
        CHECK(result[0]);
      }
      {
        INFO(description +
             "\nCheck that referenced val is updated (local space)");
        CHECK(result[1]);
      }
      {
        INFO(description + "\nError returned type (local space)");
        CHECK(result[2]);
      }
    }
  }

  bool require_combination_for_full_conformance() { return true; }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_EXCHANGE_TEST_H
