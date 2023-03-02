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
//  Provides base test class for sycl::atomic_ref api tests
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_TEST_BASE_H
#define SYCL_CTS_ATOMIC_REF_TEST_BASE_H

#include "atomic_ref_common.h"

namespace atomic_ref::tests::api {
using namespace atomic_ref::tests::common;
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomic_ref_test {
 public:
  static constexpr sycl::memory_order memory_orders[] = {
      sycl::memory_order::relaxed, sycl::memory_order::acq_rel,
      sycl::memory_order::seq_cst};
  static constexpr sycl::memory_scope memory_scopes[] = {
      sycl::memory_scope::work_item, sycl::memory_scope::sub_group,
      sycl::memory_scope::work_group, sycl::memory_scope::device,
      sycl::memory_scope::system};
  static constexpr sycl::memory_order memory_order_for_atomic_ref_obj =
      MemoryOrderT::value;
  static constexpr sycl::memory_scope memory_scope_for_atomic_ref_obj =
      MemoryScopeT::value;
  static constexpr sycl::access::address_space
      address_space_for_atomic_ref_obj = AddressSpaceT::value;

  using atomic_ref_type = sycl::atomic_ref<T, memory_order_for_atomic_ref_obj,
                                    memory_scope_for_atomic_ref_obj,
                                    address_space_for_atomic_ref_obj>;
  using ReferencedType = std::remove_pointer_t<T>;

 protected:
  ReferencedType host_data_expd;
  ReferencedType host_data_chgd;
  T host_val_expd;
  T host_val_chgd;
  sycl::queue queue;

  static constexpr bool address_space_is_not_local_space() {
    return address_space_for_atomic_ref_obj !=
           sycl::access::address_space::local_space;
  }

  static constexpr bool address_space_is_not_global_space() {
    return address_space_for_atomic_ref_obj !=
           sycl::access::address_space::global_space;
  }

  template <unsigned long result_buf_size, typename TestActionT>
  void queue_submit_local_scope(std::array<bool, result_buf_size>& result,
                                TestActionT& test_action) {
    T ref_val = host_val_expd;
    T ref_val_chgd = host_val_chgd;
    {
      sycl::buffer result_buf(result.data(), sycl::range(result_buf_size));

      queue
          .submit([&](sycl::handler& cgh) {
            auto result_accessor =
                result_buf.template get_access<sycl::access_mode::write>(cgh);
            sycl::local_accessor<T, 1> loc_acc(sycl::range<1>(1), cgh);
            cgh.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) {
              loc_acc[0] = ref_val;
              atomic_ref_type a_r(loc_acc[0]);
              test_action(ref_val, ref_val_chgd, a_r, result_accessor, loc_acc);
            });
          })
          .wait_and_throw();
    }
  }

  template <unsigned long result_buf_size, typename TestActionT>
  void queue_submit_global_scope(std::array<bool, result_buf_size>& result,
                                 TestActionT& test_action) {
    T ref_val = host_val_expd;
    T ref_val_chgd = host_val_chgd;
    {
      sycl::buffer data_buf(&ref_val, sycl::range(1));
      sycl::buffer result_buf(result.data(), sycl::range(result_buf_size));

      queue
          .submit([&](sycl::handler& cgh) {
            auto data_accessor =
                data_buf.template get_access<sycl::access_mode::read_write>(
                    cgh);
            auto result_accessor =
                result_buf.template get_access<sycl::access_mode::write>(cgh);
            cgh.single_task([=] {
              atomic_ref_type a_r(data_accessor[0]);
              test_action(ref_val, ref_val_chgd, a_r, result_accessor,
                          data_accessor);
            });
          })
          .wait_and_throw();
    }
  }

  void reset_host_values() {
    host_data_expd = value_operations::init<ReferencedType>(expected_val);
    host_data_chgd = value_operations::init<ReferencedType>(changed_val);
    if constexpr (std::is_pointer_v<T>) {
      host_val_expd = &host_data_expd;
      host_val_chgd = &host_data_chgd;
    } else {
      host_val_expd = host_data_expd;
      host_val_chgd = host_data_chgd;
    }
  }

 private:
  virtual void run_on_device(
      const std::string& type_name, const std::string& memory_order,
      const std::string& memory_scope, const std::string& address_space,
      sycl::memory_order memory_order_val = memory_order_for_atomic_ref_obj,
      sycl::memory_scope memory_scope_val =
          memory_scope_for_atomic_ref_obj) = 0;

  virtual bool require_combination_for_full_conformance() = 0;

 public:
  using sptr = std::shared_ptr<atomic_ref_test>;

  atomic_ref_test() : queue(util::get_cts_object::queue()) {
    reset_host_values();
  }
  virtual ~atomic_ref_test() = default;

  void operator()(const std::string& type_name, const std::string& memory_order,
                  const std::string& memory_scope,
                  const std::string& address_space) {
    if (memory_order_and_scope_are_supported(queue,
                                             memory_order_for_atomic_ref_obj,
                                             memory_scope_for_atomic_ref_obj)) {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
      if (require_combination_for_full_conformance()) {
        for (auto order : memory_orders) {
          for (auto scope : memory_scopes) {
            if (memory_order_and_scope_are_not_supported(queue, order, scope)) {
              continue;
            }
            run_on_device(type_name, memory_order, memory_scope, address_space,
                          order, scope);
          }
        }
      } else {
        run_on_device(type_name, memory_order, memory_scope, address_space,
                      memory_order_for_atomic_ref_obj,
                      memory_scope_for_atomic_ref_obj);
      }
#else
      run_on_device(type_name, memory_order, memory_scope, address_space,
                    memory_order_for_atomic_ref_obj,
                    memory_scope_for_atomic_ref_obj);
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
    }
  }
};

}  // namespace atomic_ref::tests::api

#endif  // SYCL_CTS_ATOMIC_REF_TEST_BASE_H
