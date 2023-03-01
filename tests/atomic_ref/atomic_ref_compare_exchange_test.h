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
//  compare_exchange_strong()/compare_exchange_weak() methods
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_COMPARE_EXCHANGE_TEST_H
#define SYCL_CTS_ATOMIC_REF_COMPARE_EXCHANGE_TEST_H

#include "atomic_ref_test_base.h"

namespace atomic_ref::tests::api {
struct strong;
struct weak;

template <typename ExchangeType, typename T, typename MemoryOrderT,
          typename MemoryScopeT, typename AddressSpaceT>
class atomic_ref_compare_exchange_test
    : public atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT> {
  using base = atomic_ref_test<T, MemoryOrderT, MemoryScopeT, AddressSpaceT>;

  using AtomicRT = typename base::AtomicRT;

  class kernel_to_check_op_with_eq_values {
   private:
    bool exec_cmpr_exch(AtomicRT& a_r, T& expected, T desired) const {
      if constexpr (std::is_same_v<ExchangeType, weak>)
        return a_r.compare_exchange_weak(expected, desired, success_order,
                                         failure_order, scope);
      else
        return a_r.compare_exchange_strong(expected, desired, success_order,
                                           failure_order, scope);
    }

    bool exec_cmpr_exch_ovrld(AtomicRT& a_r, T& expected, T desired) const {
      if constexpr (std::is_same_v<ExchangeType, weak>)
        return a_r.compare_exchange_weak(expected, desired, success_order,
                                         scope);
      else
        return a_r.compare_exchange_strong(expected, desired, success_order,
                                           scope);
    }

    template <typename res_acc_type>
    void check_cmpr_exch_weak(res_acc_type result_accessor, bool success,
                              T referenced_val, T desired,
                              int check_number) const {
      bool expd_res_of_comp_exch_op;
      if (referenced_val == desired)
        expd_res_of_comp_exch_op = true;
      else
        expd_res_of_comp_exch_op = false;
      result_accessor[check_number - 1] = success == expd_res_of_comp_exch_op;
    }

    template <typename res_acc_type>
    void check_cmpr_exch_strong(res_acc_type result_accessor, bool success,
                                T referenced_val, T desired,
                                int check_number) const {
      result_accessor[check_number - 1] = success == true;
      result_accessor[check_number + 1] = referenced_val == desired;
    }

    template <typename res_acc_type>
    void check_cmpr_exch(res_acc_type result_accessor, bool success,
                         T referenced_val, T desired, int check_number) const {
      if constexpr (std::is_same_v<ExchangeType, weak>)
        check_cmpr_exch_weak(result_accessor, success, referenced_val, desired,
                             check_number);
      else
        check_cmpr_exch_strong(result_accessor, success, referenced_val,
                               desired, check_number);
    }
    sycl::memory_order success_order;
    sycl::memory_order failure_order;
    sycl::memory_scope scope;

   public:
    kernel_to_check_op_with_eq_values(sycl::memory_order success_order,
                                      sycl::memory_order failure_order,
                                      sycl::memory_scope scope) {
      this->success_order = success_order;
      this->failure_order = failure_order;
      this->scope = scope;
    }

    template <typename res_acc_type, typename data_acc_type>
    void operator()(T ref_val, T ref_val_chgd, AtomicRT& a_r,
                    res_acc_type res_accessor,
                    data_acc_type data_accessor) const {
      T expected = ref_val;
      T desired = ref_val_chgd;

      bool success = exec_cmpr_exch(a_r, expected, desired);
      int check_number = 1;

      check_cmpr_exch(res_accessor, success, data_accessor[0], desired,
                      check_number++);

      data_accessor[0] = expected;

      success = exec_cmpr_exch_ovrld(a_r, expected, desired);

      check_cmpr_exch(res_accessor, success, data_accessor[0], desired,
                      check_number);
    }
  };
  class kernel_to_check_op_with_uneq_values {
   private:
    bool exec_cmpr_exch(AtomicRT& a_r, T& expected, T desired) const {
      if constexpr (std::is_same_v<ExchangeType, weak>)
        return a_r.compare_exchange_weak(expected, desired, success_order,
                                         failure_order, scope);
      else
        return a_r.compare_exchange_strong(expected, desired, success_order,
                                           failure_order, scope);
    }

    bool exec_cmpr_exch_ovrld(AtomicRT& a_r, T& expected, T desired) const {
      if constexpr (std::is_same_v<ExchangeType, weak>)
        return a_r.compare_exchange_weak(expected, desired, success_order,
                                         scope);
      else
        return a_r.compare_exchange_strong(expected, desired, success_order,
                                           scope);
    }
    sycl::memory_order success_order;
    sycl::memory_order failure_order;
    sycl::memory_scope scope;

   public:
    kernel_to_check_op_with_uneq_values(sycl::memory_order success_order,
                                        sycl::memory_order failure_order,
                                        sycl::memory_scope scope) {
      this->success_order = success_order;
      this->failure_order = failure_order;
      this->scope = scope;
    }

    template <typename res_acc_type, typename data_acc_type>
    void operator()(T ref_val, T ref_val_chgd, AtomicRT& a_r,
                    res_acc_type res_accessor,
                    data_acc_type data_accessor) const {
      T expected = ref_val_chgd;
      T desired = ref_val;

      auto success = exec_cmpr_exch(a_r, expected, desired);

      res_accessor[0] = success == false;
      res_accessor[1] = expected == data_accessor[0];

      expected = ref_val_chgd;

      auto another_success = exec_cmpr_exch_ovrld(a_r, expected, desired);

      res_accessor[2] = another_success == false;
      res_accessor[3] = expected == data_accessor[0];

      res_accessor[4] = std::is_same_v<decltype(success), bool>;
      res_accessor[5] = std::is_same_v<decltype(another_success), bool>;
    }
  };

  std::string checked_method_name;
  std::string test_description;
  sycl::memory_order memory_order_read_write;
  sycl::memory_order memory_order_read;
  sycl::memory_scope memory_scope_val;

  void check_for_equal_values_on_device();
  void check_for_unequal_values_on_device();
  void check_comp_exch_result_for_eq_vals(std::array<bool, 4>& result);
  void check_comp_exch_result_for_uneq_vals(std::array<bool, 6>& result);

 public:
  atomic_ref_compare_exchange_test();

 private:
  void run_on_device(const std::string& type_name,
                     const std::string& memory_order,
                     const std::string& memory_scope,
                     const std::string& address_space,
                     sycl::memory_order memory_order_val =
                         base::memory_order_for_atomic_ref_obj,
                     sycl::memory_scope memory_scope_val =
                         base::memory_scope_for_atomic_ref_obj);

  bool require_combination_for_full_conformance() { return true; }
};

template <typename ExchangeType, typename T, typename MemoryOrderT,
          typename MemoryScopeT, typename AddressSpaceT>
atomic_ref_compare_exchange_test<
    ExchangeType, T, MemoryOrderT, MemoryScopeT,
    AddressSpaceT>::atomic_ref_compare_exchange_test() {
  if constexpr (std::is_same_v<ExchangeType, weak>) {
    checked_method_name = "Check compare_exchange_weak() method";
  } else {
    checked_method_name = "Check compare_exchange_strong() method";
  }
}

template <typename ExchangeType, typename T, typename MemoryOrderT,
          typename MemoryScopeT, typename AddressSpaceT>
void atomic_ref_compare_exchange_test<
    ExchangeType, T, MemoryOrderT, MemoryScopeT,
    AddressSpaceT>::run_on_device(const std::string& type_name,
                                  const std::string& memory_order,
                                  const std::string& memory_scope,
                                  const std::string& address_space,
                                  sycl::memory_order memory_order_val,
                                  sycl::memory_scope memory_scope_val) {
  test_description =
      get_section_name(type_name, memory_order, memory_scope, address_space,
                       memory_order_val, memory_scope_val, checked_method_name);
  memory_order_read_write = memory_order_val;
  memory_order_read = memory_order_val == sycl::memory_order::acq_rel
                          ? sycl::memory_order::acquire
                          : memory_order_val;
  this->memory_scope_val = memory_scope_val;
  check_for_equal_values_on_device();
  check_for_unequal_values_on_device();
}

template <typename ExchangeType, typename T, typename MemoryOrderT,
          typename MemoryScopeT, typename AddressSpaceT>
void atomic_ref_compare_exchange_test<
    ExchangeType, T, MemoryOrderT, MemoryScopeT,
    AddressSpaceT>::check_for_equal_values_on_device() {
  kernel_to_check_op_with_eq_values comp_exch_test{
      memory_order_read_write, memory_order_read, memory_scope_val};

  if constexpr (base::address_space_is_not_local_space()) {
    std::array result{false, false, false, false};
    this->queue_submit_global_scope(result, comp_exch_test);
    check_comp_exch_result_for_eq_vals(result);
  }

  if constexpr (base::address_space_is_not_global_space()) {
    std::array result{false, false, false, false};
    this->queue_submit_local_scope(result, comp_exch_test);
    check_comp_exch_result_for_eq_vals(result);
  }
}

template <typename ExchangeType, typename T, typename MemoryOrderT,
          typename MemoryScopeT, typename AddressSpaceT>
void atomic_ref_compare_exchange_test<ExchangeType, T, MemoryOrderT,
                                      MemoryScopeT, AddressSpaceT>::
    check_comp_exch_result_for_eq_vals(std::array<bool, 4>& result) {
  {
    INFO(test_description + "\ncompare_exchange call failed");
    CHECK(result[0]);
  }
  {
    INFO(test_description + "\ncompare_exchange_overloaded call failed");
    CHECK(result[1]);
  }
  if constexpr (std::is_same_v<ExchangeType, strong>) {
    {
      INFO(test_description +
           "\nError, referenced value is not updated after compare_exchange "
           "call "
           "with equal values");
      CHECK(result[2]);
    }
    {
      INFO(test_description +
           "\nError, referenced value is not updated after "
           "compare_exchange_overloaded call with equal values");
      CHECK(result[3]);
    }
  }
}

template <typename ExchangeType, typename T, typename MemoryOrderT,
          typename MemoryScopeT, typename AddressSpaceT>
void atomic_ref_compare_exchange_test<
    ExchangeType, T, MemoryOrderT, MemoryScopeT,
    AddressSpaceT>::check_for_unequal_values_on_device() {
  kernel_to_check_op_with_uneq_values comp_exch_test{
      memory_order_read_write, memory_order_read, memory_scope_val};

  if constexpr (base::address_space_is_not_local_space()) {
    std::array result{false, false, false, false, false, false};
    this->queue_submit_global_scope(result, comp_exch_test);
    check_comp_exch_result_for_uneq_vals(result);
  }

  if constexpr (base::address_space_is_not_global_space()) {
    std::array result{false, false, false, false, false, false};
    this->queue_submit_local_scope(result, comp_exch_test);
    check_comp_exch_result_for_uneq_vals(result);
  }
}

template <typename ExchangeType, typename T, typename MemoryOrderT,
          typename MemoryScopeT, typename AddressSpaceT>
void atomic_ref_compare_exchange_test<ExchangeType, T, MemoryOrderT,
                                      MemoryScopeT, AddressSpaceT>::
    check_comp_exch_result_for_uneq_vals(std::array<bool, 6>& result) {
  {
    INFO(test_description +
         "\nError, compare_exchange call with uneq values updated "
         "referenced value");
    CHECK(result[0]);
  }
  {
    INFO(test_description +
         "\nError, \"expected\" argument value is not upadted after "
         "compare_exchange call with uneq values");
    CHECK(result[1]);
  }
  {
    INFO(test_description +
         "\nError, compare_exchange_overloaded call with uneq values"
         " updated referenced value");
    CHECK(result[2]);
  }
  {
    INFO(test_description +
         "\nError, \"expected\" argument value is not upadted after "
         "compare_exchange_overloaded call with uneq values");
    CHECK(result[3]);
  }
  {
    INFO(test_description + "\nError returned type for compare_exchange()");
    CHECK(result[4]);
  }
  {
    INFO(test_description +
         "\nError returned type for compare_exchange_overloaded()");
    CHECK(result[5]);
  }
}

template <typename T>
struct run_compare_exchange_test {
  void operator()(const std::string& type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<atomic_ref_compare_exchange_test, weak, T>(
        memory_orders, memory_scopes, address_spaces, type_name);

    std::string type_name_for_pointer_types = type_name + "*";
    for_all_combinations<atomic_ref_compare_exchange_test, weak, T*>(
        memory_orders, memory_scopes, address_spaces,
        type_name_for_pointer_types);

    for_all_combinations<atomic_ref_compare_exchange_test, strong, T>(
        memory_orders, memory_scopes, address_spaces, type_name);

    for_all_combinations<atomic_ref_compare_exchange_test, strong, T*>(
        memory_orders, memory_scopes, address_spaces,
        type_name_for_pointer_types);
  }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_COMPARE_EXCHANGE_TEST_H
