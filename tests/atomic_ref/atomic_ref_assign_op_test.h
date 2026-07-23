/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::atomic_ref::operator=() member function
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_ASSIGN_OP_TEST_H
#define SYCL_CTS_ATOMIC_REF_ASSIGN_OP_TEST_H

#include "atomic_ref_test_base.h"

namespace atomic_ref::tests::api {
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomic_ref_assign_op_test
    : public atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT> {
  using base = atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT>;

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
                         "Check if operator=() stores \"desired\" to the object"
                         " referenced by this atomic_ref and returned value is "
                         "\"desired\" in device code");
    auto assign_op_test = [](T val_expd, T val_chgd,
                             typename base::atomic_ref_type& a_r,
                             auto result_acc, auto ref_data_acc) {
      auto desired = (a_r = val_chgd);
      result_acc[0] = (ref_data_acc[0] == val_chgd);
      result_acc[1] = (desired == val_chgd);
      result_acc[2] = std::is_same_v<decltype(desired), T>;
    };

    if constexpr (base::address_space_is_not_local_space()) {
      std::array result{false, false, false};
      this->queue_submit_global_scope(result, assign_op_test);
      {
        INFO(
            description +
            "\nError, call of operator=() didn't update referenced val (global "
            "space)");
        CHECK(result[0]);
      }
      {
        INFO(description +
             "\nError returned value of operator=() (global space)");
        CHECK(result[1]);
      }
      {
        INFO(description +
             "\nError returned type of operator=() (global space)");
        CHECK(result[2]);
      }
    }

    if constexpr (base::address_space_is_not_global_space()) {
      std::array result{false, false, false};
      this->queue_submit_local_scope(result, assign_op_test);
      {
        INFO(description +
             "\nError, call of operator=() didn't update referenced val (local "
             "space)");
        CHECK(result[0]);
      }
      {
        INFO(description +
             "\nError returned value of operator=() (local space)");
        CHECK(result[1]);
      }
      {
        INFO(description +
             "\nError returned type of operator=() (local space)");
        CHECK(result[2]);
      }
    }
  }

  bool require_combination_for_full_conformance() { return false; }
};

template <typename T>
struct run_assign_op_test {
  void operator()(const std::string& type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<atomic_ref_assign_op_test, T>(
        memory_orders, memory_scopes, address_spaces, type_name);
  }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_ASSIGN_OP_TEST_H
