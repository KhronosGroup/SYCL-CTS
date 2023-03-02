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
//  Provides tests for sycl::atomic_ref fetch_min()/fetch_max() methods
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_FETCH_MIN_MAX_TEST_H
#define SYCL_CTS_ATOMIC_REF_FETCH_MIN_MAX_TEST_H

#include "atomic_ref_test_base.h"

namespace atomic_ref::tests::api {
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomic_ref_fetch_min_max_test
    : public atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT> {
  using base = atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT>;
  T big_value;
  T small_value;

  void check_test_result_buffer(std::array<bool, 10>& result,
                                const std::string& description,
                                std::string addr_space) {
    {
      INFO(description + "\nError returned val for fetch_max (" + addr_space +
           "space)");
      CHECK(result[0]);
    }
    {
      INFO(description +
           "\nError, referenced val is not updated after fetch_max (" +
           addr_space + " space)");
      CHECK(result[1]);
    }
    {
      INFO(description + "\nError returned val for fetch_min (" + addr_space +
           " space)");
      CHECK(result[2]);
    }
    {
      INFO(description +
           "\nError, referenced val is not updated after fetch_min (" +
           addr_space + " space)");
      CHECK(result[3]);
    }
    {
      INFO(description +
           "\nError returned val for fetch_min with operand value grater than "
           "referenced val (" +
           addr_space + " space)");
      CHECK(result[4]);
    }
    {
      INFO(description +
           "\nError, referenced val is updated after fetch_min with "
           "operand value grater than referenced val (" +
           addr_space + " space)");
      CHECK(result[5]);
    }
    {
      INFO(description +
           "\nError returned val for fetch_max with operand value less than "
           "referenced val (" +
           addr_space + " space)");
      CHECK(result[6]);
    }
    {
      INFO(description +
           "\nError, referenced val is updated after fetch_max with "
           "operand value less than referenced val (" +
           addr_space + " space)");
      CHECK(result[7]);
    }
    {
      INFO(description + "\nError returned type for fetch_max() (" +
           addr_space + " space)");
      CHECK(result[8]);
    }
    {
      INFO(description + "\nError returned type for fetch_min() (" +
           addr_space + " space)");
      CHECK(result[9]);
    }
  }

 public:
  atomic_ref_fetch_min_max_test() {
    if constexpr (std::is_floating_point_v<T>) {
      big_value = this->host_val_expd + 1.25;
      small_value = this->host_val_expd - 1.25;
    } else {
      big_value = this->host_val_expd + 1;
      small_value = this->host_val_expd - 1;
    }
  }

 private:
  bool require_combination_for_full_conformance() { return true; }

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
                         "Check if fetch_min()/fetch_max() method compute "
                         "minimum or maximum of operand"
                         " and the value of the referenced object, assign "
                         "result to the referenced object"
                         " and returns the original value of "
                         " the referenced object in device code");
    T big_value_copy = big_value;
    T small_value_copy = small_value;
    auto fetch_min_max_test = [memory_order_val, memory_scope_val,
                               big_value_copy, small_value_copy](
                                  T val_expd, T val_chgd,
                                  typename base::atomic_ref_type& a_r, auto result_acc,
                                  auto ref_data_acc) {
      T original_val = ref_data_acc[0];

      auto ref_val_before_fetch_max =
          a_r.fetch_max(big_value_copy, memory_order_val, memory_scope_val);

      result_acc[0] = ref_val_before_fetch_max == original_val;
      if constexpr (std::is_floating_point_v<T>)
        result_acc[1] = compare_floats(ref_data_acc[0], big_value_copy);
      else
        result_acc[1] = ref_data_acc[0] == big_value_copy;

      auto ref_val_before_fetch_min =
          a_r.fetch_min(original_val, memory_order_val, memory_scope_val);

      result_acc[2] = ref_val_before_fetch_min == big_value_copy;
      if constexpr (std::is_floating_point_v<T>)
        result_acc[3] = compare_floats(ref_data_acc[0], original_val);
      else
        result_acc[3] = ref_data_acc[0] == original_val;

      auto ref_val_before_fetch_min_with_operand_grater_than_ref_value =
          a_r.fetch_min(big_value_copy, memory_order_val, memory_scope_val);

      result_acc[4] =
          ref_val_before_fetch_min_with_operand_grater_than_ref_value ==
          original_val;
      if constexpr (std::is_floating_point_v<T>)
        result_acc[5] = compare_floats(ref_data_acc[0], original_val);
      else
        result_acc[5] = ref_data_acc[0] == original_val;

      auto ref_val_before_fetch_max_with_operand_less_than_ref_value =
          a_r.fetch_max(small_value_copy, memory_order_val, memory_scope_val);

      result_acc[6] =
          ref_val_before_fetch_max_with_operand_less_than_ref_value ==
          original_val;
      if constexpr (std::is_floating_point_v<T>)
        result_acc[7] = compare_floats(ref_data_acc[0], original_val);
      else
        result_acc[7] = ref_data_acc[0] == original_val;

      result_acc[8] = std::is_same_v<decltype(ref_val_before_fetch_max), T>;
      result_acc[9] = std::is_same_v<decltype(ref_val_before_fetch_min), T>;
    };

    if constexpr (base::address_space_is_not_local_space()) {
      std::array result{false, false, false, false, false,
                        false, false, false, false, false};
      this->queue_submit_global_scope(result, fetch_min_max_test);
      check_test_result_buffer(result, description, "global");
    }

    if constexpr (base::address_space_is_not_global_space()) {
      std::array result{false, false, false, false, false,
                        false, false, false, false, false};
      this->queue_submit_local_scope(result, fetch_min_max_test);
      check_test_result_buffer(result, description, "local");
    }
  }
};

template <typename T>
struct run_fetch_min_max_test {
  void operator()(const std::string& type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<atomic_ref_fetch_min_max_test, T>(
        memory_orders, memory_scopes, address_spaces, type_name);
  }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_FETCH_MIN_MAX_TEST_H
