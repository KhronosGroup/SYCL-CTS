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
//  Provides tests for sycl::atomic_ref operator+=()/operator-=() methods
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_ADD_SUB_OP_ALL_TYPES_TEST_H
#define SYCL_CTS_ATOMIC_REF_ADD_SUB_OP_ALL_TYPES_TEST_H

#include "atomic_ref_test_base.h"

namespace atomic_ref::tests::api {
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomic_ref_add_sub_op_all_types_test
    : public atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT> {
  using base = atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT>;

  using operand_type = std::conditional_t<std::is_pointer_v<T>, ptrdiff_t, T>;

  operand_type operand_val;

  void check_test_result_buffer(std::array<bool, 6>& result,
                                const std::string& description,
                                std::string addr_space) {
    {
      INFO(description + "\nError returned val for add (" + addr_space +
           " space)");
      CHECK(result[0]);
    }
    {
      INFO(description + "\nError, referenced val is not updated after add (" +
           addr_space + " space)");
      CHECK(result[1]);
    }
    {
      INFO(description + "\nError returned val for subtract (" + addr_space +
           " space)");
      CHECK(result[2]);
    }
    {
      INFO(description +
           "\nError, referenced val is not updated after subtract (" +
           addr_space + " space)");
      CHECK(result[3]);
    }
    {
      INFO(description + "\nError returned type for add (" + addr_space +
           " space)");
      CHECK(result[4]);
    }
    {
      INFO(description + "\nError returned type for subtract (" + addr_space +
           " space)");
      CHECK(result[5]);
    }
  }

 public:
  atomic_ref_add_sub_op_all_types_test() {
    if constexpr (std::is_floating_point_v<T>)
      operand_val = 1.25;
    else
      operand_val = 1;
  }

 private:
  void run_on_device(const std::string& type_name,
                     const std::string& memory_order,
                     const std::string& memory_scope,
                     const std::string& address_space,
                     sycl::memory_order memory_order_val =
                         base::memory_order_for_atomic_ref_obj,
                     sycl::memory_scope memory_scope_val =
                         base::memory_scope_for_atomic_ref_obj) {
    std::string description = get_section_name(
        type_name, memory_order, memory_scope, address_space,
        "Check if operator+=()/operator-=() adds/subtract the operand to the "
        "object referenced by this atomic_ref"
        " and returns the original value +/- operand"
        " in device code");
    operand_type operand_val_copy = operand_val;
    auto add_sub_op_test = [operand_val_copy](
                               T val_expd, T val_chgd,
                               typename base::atomic_ref_type& a_r,
                               auto result_acc, auto ref_data_acc) {
      T original_val = val_expd;
      T val_expd_after_adding = original_val + operand_val_copy;
      T val_expd_after_subtract = original_val;

      auto returned_value_after_add = (a_r += operand_val_copy);

      result_acc[0] = returned_value_after_add == val_expd_after_adding;
      if constexpr (std::is_floating_point_v<T>)
        result_acc[1] = compare_floats(ref_data_acc[0], val_expd_after_adding);
      else
        result_acc[1] = ref_data_acc[0] == val_expd_after_adding;

      auto returned_value_after_subtract = (a_r -= operand_val_copy);

      result_acc[2] = returned_value_after_subtract == val_expd_after_subtract;
      if constexpr (std::is_floating_point_v<T>)
        result_acc[3] =
            compare_floats(ref_data_acc[0], val_expd_after_subtract);
      else
        result_acc[3] = ref_data_acc[0] == val_expd_after_subtract;

      result_acc[4] = std::is_same_v<decltype(returned_value_after_add), T>;
      result_acc[5] =
          std::is_same_v<decltype(returned_value_after_subtract), T>;
    };

    if constexpr (base::address_space_is_not_local_space()) {
      std::array result{false, false, false, false, false, false};
      this->queue_submit_global_scope(result, add_sub_op_test);
      check_test_result_buffer(result, description, "global");
    }

    if constexpr (base::address_space_is_not_global_space()) {
      std::array result{false, false, false, false, false, false};
      this->queue_submit_local_scope(result, add_sub_op_test);
      check_test_result_buffer(result, description, "local");
    }
  }

  bool require_combination_for_full_conformance() { return false; }
};

template <typename T>
struct run_add_sub_op_all_types_test {
  void operator()(const std::string& type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<atomic_ref_add_sub_op_all_types_test, T>(
        memory_orders, memory_scopes, address_spaces, type_name);
  }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_ADD_SUB_OP_ALL_TYPES_TEST_H
