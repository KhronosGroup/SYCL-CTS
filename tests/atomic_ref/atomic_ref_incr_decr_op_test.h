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
//  Provides tests for sycl::atomic_ref
//  operator++(int)/operator++()/operator--(int)/operator--() methods
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_INCR_DECR_OP_TEST_H
#define SYCL_CTS_ATOMIC_REF_INCR_DECR_OP_TEST_H

#include "atomic_ref_test_base.h"

namespace atomic_ref::tests::api {
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomic_ref_incr_decr_op_test
    : public atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT> {
  using base = atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT>;

  void check_test_result_buffer(std::array<bool, 12>& result,
                                const std::string& description,
                                std::string addr_space) {
    {
      INFO(description + "\nError returned val for post incr op (" +
           addr_space + " space)");
      CHECK(result[0]);
    }
    {
      INFO(description +
           "\nReferenced val is not updated after post incr op (" + addr_space +
           " space)");
      CHECK(result[1]);
    }
    {
      INFO(description + "\nError returned val for prfx incr op (" +
           addr_space + " space)");
      CHECK(result[2]);
    }
    {
      INFO(description +
           "\nReferenced val is not updated after prfx incr op (" + addr_space +
           " space)");
      CHECK(result[3]);
    }
    {
      INFO(description + "\nError returned val for post decr op (" +
           addr_space + " space)");
      CHECK(result[4]);
    }
    {
      INFO(description +
           "\nReferenced val is not updated after post decr op (" + addr_space +
           " space)");
      CHECK(result[5]);
    }
    {
      INFO(description + "\nError returned val for prfx decr op (" +
           addr_space + " space)");
      CHECK(result[6]);
    }
    {
      INFO(description +
           "\nReferenced val is not updated after prfx decr op (" + addr_space +
           " space)");
      CHECK(result[7]);
    }
    {
      INFO(description + "\nError returned type for postfix ++ (" + addr_space +
           " space)");
      CHECK(result[8]);
    }
    {
      INFO(description + "\nError returned type for prefix ++ (" + addr_space +
           " space)");
      CHECK(result[9]);
    }
    {
      INFO(description + "\nError returned type for postfix -- (" + addr_space +
           " space)");
      CHECK(result[10]);
    }
    {
      INFO(description + "\nError returned type for prefix -- (" + addr_space +
           " space)");
      CHECK(result[11]);
    }
  }

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
                         "Check increment/decrement operators in device code");
    auto incr_op_test = [](T val_expd, T val_chgd,
                           typename base::atomic_ref_type& a_r, auto result_acc,
                           auto ref_data_acc) {
      T val_before_op = ref_data_acc[0];

      auto ref_val_before_post_incr = a_r++;

      result_acc[0] = ref_val_before_post_incr == val_before_op;
      result_acc[1] = ref_data_acc[0] == val_before_op + 1;

      auto ref_val_after_prfx_incr = ++a_r;

      result_acc[2] = ref_val_after_prfx_incr == val_before_op + 2;
      result_acc[3] = ref_data_acc[0] == val_before_op + 2;

      auto ref_val_before_post_decr = a_r--;

      result_acc[4] = ref_val_before_post_decr == val_before_op + 2;
      result_acc[5] = ref_data_acc[0] == val_before_op + 1;

      auto ref_val_after_prfx_decr = --a_r;

      result_acc[6] = ref_val_after_prfx_decr == val_before_op;
      result_acc[7] = ref_data_acc[0] == val_before_op;

      result_acc[8] = std::is_same_v<decltype(ref_val_before_post_incr), T>;
      result_acc[9] = std::is_same_v<decltype(ref_val_after_prfx_incr), T>;
      result_acc[10] = std::is_same_v<decltype(ref_val_before_post_decr), T>;
      result_acc[11] = std::is_same_v<decltype(ref_val_after_prfx_decr), T>;
    };

    if constexpr (base::address_space_is_not_local_space()) {
      std::array result = {false, false, false, false, false, false,
                           false, false, false, false, false, false};
      this->queue_submit_global_scope(result, incr_op_test);
      check_test_result_buffer(result, description, "global");
    }

    if constexpr (base::address_space_is_not_global_space()) {
      std::array result = {false, false, false, false, false, false,
                           false, false, false, false, false, false};
      this->queue_submit_local_scope(result, incr_op_test);
      check_test_result_buffer(result, description, "local");
    }
  }

  bool require_combination_for_full_conformance() { return false; }
};

template <typename T>
struct run_incr_decr_op_test {
  void operator()(const std::string& type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    if constexpr (std::is_integral_v<T>) {
      for_all_combinations<atomic_ref_incr_decr_op_test, T>(
          memory_orders, memory_scopes, address_spaces, type_name);
    }

    if (is_64_bits_pointer<T*>() && device_has_not_aspect_atomic64()) return;

    std::string type_name_for_pointer_types = type_name + "*";
    for_all_combinations<atomic_ref_incr_decr_op_test, T*>(
        memory_orders, memory_scopes, address_spaces,
        type_name_for_pointer_types);
  }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_INCR_DECR_OP_TEST_H
