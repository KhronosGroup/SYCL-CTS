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
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_STRESS_TEST_H
#define SYCL_CTS_ATOMIC_REF_STRESS_TEST_H

#include "../atomic_ref/atomic_ref_common.h"
#include "../common/once_per_unit.h"
#include "../common/section_name_builder.h"
#include "../common/type_coverage.h"

#include <atomic>

constexpr size_t max_size = 32768;

namespace atomic_ref_stress_test {
using namespace sycl_cts;

template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomicity_device_scope {
  static constexpr sycl::memory_order MemoryOrder = MemoryOrderT::value;
  static constexpr sycl::memory_scope MemoryScope = MemoryScopeT::value;
  static constexpr sycl::access::address_space AddressSpace =
      AddressSpaceT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& memory_order_name,
                  const std::string& memory_scope_name,
                  const std::string& address_space_name) {
    INFO(atomic_ref::tests::common::get_section_name(
        type_name, memory_order_name, memory_scope_name, address_space_name,
        "atomicity_device_scope"));
    auto queue = once_per_unit::get_queue();
    if (!atomic_ref::tests::common::memory_order_and_scope_are_supported(
            queue, MemoryOrder, MemoryScope))
      return;
    T val{};
    const size_t size = std::min<size_t>(
        queue.get_device().get_info<sycl::info::device::max_compute_units>(),
        max_size);
    {
      sycl::buffer buf{&val, {1}};
      queue.submit([&](sycl::handler& cgh) {
        sycl::accessor acc{buf, cgh};
        cgh.parallel_for({size}, [=](auto i) {
          sycl::atomic_ref<T, MemoryOrder, MemoryScope, AddressSpace> a_r{
              acc[0]};
          a_r.fetch_add(2);
        });
      });
    }
    bool res;
    if constexpr (std::is_floating_point_v<T>)
      res = atomic_ref::tests::common::compare_floats<T>(val, size * 2);
    else
      res = (val == size * 2);
    CHECK(res);
  }
};

template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomicity_work_group_scope {
  static constexpr sycl::memory_order MemoryOrder = MemoryOrderT::value;
  static constexpr sycl::memory_scope MemoryScope = MemoryScopeT::value;
  static constexpr sycl::access::address_space AddressSpace =
      AddressSpaceT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& memory_order_name,
                  const std::string& memory_scope_name,
                  const std::string& address_space_name) {
    INFO(atomic_ref::tests::common::get_section_name(
        type_name, memory_order_name, memory_scope_name, address_space_name,
        "atomicity_work_group_scope"));
    auto queue = once_per_unit::get_queue();
    if (!atomic_ref::tests::common::memory_order_and_scope_are_supported(
            queue, MemoryOrder, MemoryScope))
      return;
    constexpr size_t group_range = 4;
    const size_t local_range = std::min<size_t>(
        queue.get_device().get_info<sycl::info::device::max_work_group_size>(),
        max_size);
    std::array<T, group_range> vals;
    vals.fill(0);
    {
      sycl::buffer buf{vals.data(), {group_range}};
      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor acc{buf, cgh};
            sycl::local_accessor<T> lacc{{1}, cgh};
            cgh.parallel_for(
                sycl::nd_range<1>(group_range * local_range, local_range),
                [=](auto item) {
                  sycl::atomic_ref<T, MemoryOrder, MemoryScope, AddressSpace>
                      a_r{lacc[0]};
                  a_r.store(0);
                  sycl::group_barrier(item.get_group());
                  if (a_r.fetch_sub(T(2)) - T(2) == -T(local_range * 2)) {
                    acc[item.get_group_linear_id()] = a_r.load();
                  }
                });
          })
          .wait_and_throw();
    }
    CHECK(std::all_of(vals.cbegin(), vals.cend(), [=](T i) {
      if constexpr (std::is_floating_point_v<T>)
        return atomic_ref::tests::common::compare_floats(i,
                                                         -T(local_range * 2));
      else
        return i == -T(local_range * 2);
    }));
  }
};

template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class aquire_release {
  static constexpr sycl::memory_order MemoryOrder = MemoryOrderT::value;
  static constexpr sycl::memory_scope MemoryScope = MemoryScopeT::value;
  static constexpr sycl::access::address_space AddressSpace =
      AddressSpaceT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& memory_order_name,
                  const std::string& memory_scope_name,
                  const std::string& address_space_name) {
    INFO(atomic_ref::tests::common::get_section_name(
        type_name, memory_order_name, memory_scope_name, address_space_name,
        "aquire_release"));
    auto queue = once_per_unit::get_queue();
    if (!atomic_ref::tests::common::memory_order_and_scope_are_supported(
            queue, MemoryOrder, MemoryScope))
      return;
    constexpr size_t global_range = 64;
    constexpr size_t local_range = 2;
    sycl::nd_range<1> nd_range(global_range, local_range);
    std::array<bool, global_range / local_range> res;
    res.fill(false);
    {
      sycl::buffer buf{res.data(), {global_range / local_range}};
      queue.submit([&](sycl::handler& cgh) {
        sycl::accessor res_acc{buf, cgh};
        sycl::local_accessor<T, 0> x{cgh};
        sycl::local_accessor<T, 0> y{cgh};
        sycl::local_accessor<T, 0> A{cgh};
        sycl::local_accessor<T, 0> B{cgh};
        cgh.parallel_for(nd_range, [=](auto item) {
          sycl::atomic_ref<T, MemoryOrder, MemoryScope, AddressSpace> refA{A};
          sycl::atomic_ref<T, MemoryOrder, MemoryScope, AddressSpace> refB{B};
          refA.store(0);
          refB.store(0);
          sycl::group_barrier(item.get_group());
          if (item.get_local_id() == sycl::id(0)) {
            x = refA.load();
            refB.store(1);
          } else {
            y = refB.load();
            refA.store(1);
          }
          sycl::group_barrier(item.get_group());
          if (item.get_local_id() == sycl::id(0))
            res_acc[item.get_group_linear_id()] = !(x == 1 && y == 1);
        });
      });
    }
    CHECK(std::all_of(res.cbegin(), res.cend(), [=](T i) { return i; }));
  }
};

template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class ordering {
  static constexpr sycl::memory_order MemoryOrder1 = MemoryOrderT::value;
  static constexpr sycl::memory_order MemoryOrder2 =
      (MemoryOrder1 == sycl::memory_order::release)
          ? sycl::memory_order::acquire
          : sycl::memory_order::seq_cst;
  static constexpr sycl::memory_scope MemoryScope = MemoryScopeT::value;
  static constexpr sycl::access::address_space AddressSpace =
      AddressSpaceT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& memory_order_name,
                  const std::string& memory_scope_name,
                  const std::string& address_space_name) {
    INFO(atomic_ref::tests::common::get_section_name(
        type_name, memory_order_name, memory_scope_name, address_space_name,
        "ordering"));
    auto queue = once_per_unit::get_queue();
    if (!atomic_ref::tests::common::memory_order_and_scope_are_supported(
            queue, MemoryOrder1, MemoryScope))
      return;
    if (!atomic_ref::tests::common::memory_order_is_supported(queue,
                                                              MemoryOrder2))
      return;
    size_t local_range =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    constexpr size_t global_range = max_size;
    if (local_range > global_range) local_range = global_range;
    sycl::nd_range<1> nd_range(global_range, local_range);
    std::array<bool, global_range> res;
    res.fill(false);
    {
      sycl::buffer buf{res.data(), {global_range}};
      queue.submit([&](sycl::handler& cgh) {
        sycl::accessor res_acc{buf, cgh};
        sycl::local_accessor<T, 0> local_acc{cgh};
        sycl::local_accessor<bool> arr_acc{{local_range}, cgh};
        cgh.parallel_for(nd_range, [=](auto item) {
          arr_acc[item.get_local_id()] = false;
          sycl::group_barrier(item.get_group());
          sycl::atomic_ref<T, sycl::memory_order::relaxed, MemoryScope,
                           AddressSpace>
              a_r{local_acc};
          arr_acc[item.get_local_id()] = true;
          a_r.store(item.get_local_id(), MemoryOrder1);
          res_acc[item.get_global_id()] =
              (arr_acc[a_r.load(MemoryOrder2)] == true);
        });
      });
    }
    CHECK(std::all_of(res.cbegin(), res.cend(), [=](T i) { return i; }));
  }
};
#ifdef __cpp_lib_atomic_ref
template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class atomicity_with_host_code {
  static constexpr sycl::memory_order MemoryOrder = MemoryOrderT::value;
  static constexpr sycl::memory_scope MemoryScope = MemoryScopeT::value;
  static constexpr sycl::access::address_space AddressSpace =
      AddressSpaceT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& memory_order_name,
                  const std::string& memory_scope_name,
                  const std::string& address_space_name) {
    INFO(atomic_ref::tests::common::get_section_name(
        type_name, memory_order_name, memory_scope_name, address_space_name,
        "atomicity_with_host_code"));
    auto queue = once_per_unit::get_queue();
    if (!atomic_ref::tests::common::memory_order_is_supported(queue,
                                                              MemoryOrder))
      return;
    if (!atomic_ref::tests::common::memory_scope_is_supported(
            queue, sycl::memory_scope::system))
      return;
    const size_t size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    const size_t count = size * 2;
    T* pval = sycl::malloc_shared<T>(1, queue);
    *pval = 0;
    std::atomic_ref<T> a_host{*pval};
    auto event = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for({size}, [=](auto i) {
        sycl::atomic_ref<T, MemoryOrder, sycl::memory_scope::system,
                         AddressSpace>
            a_dev{*pval};
        a_dev.fetch_add(1);
      });
    });
    for (int i = 0; i < count; i++) a_host.fetch_add(1);
    event.wait();
    CHECK(*pval == size + count);
  }
};
#endif
template <typename T>
struct run_atomicity_device_scope {
  void operator()(const std::string& type_name) {
    const auto memory_orders =
        value_pack<sycl::memory_order, sycl::memory_order::relaxed,
                   sycl::memory_order::acq_rel,
                   sycl::memory_order::seq_cst>::generate_named();
    const auto memory_scopes =
        value_pack<sycl::memory_scope, sycl::memory_scope::device,
                   sycl::memory_scope::system>::generate_named();
    const auto address_spaces = value_pack<
        sycl::access::address_space, sycl::access::address_space::global_space,
        sycl::access::address_space::generic_space>::generate_named();

    for_all_combinations<atomicity_device_scope, T>(
        memory_orders, memory_scopes, address_spaces, type_name);
  }
};

template <typename T>
struct run_atomicity_work_group_scope {
  void operator()(const std::string& type_name) {
    const auto memory_orders =
        value_pack<sycl::memory_order, sycl::memory_order::relaxed,
                   sycl::memory_order::acq_rel,
                   sycl::memory_order::seq_cst>::generate_named();
    const auto memory_scopes =
        value_pack<sycl::memory_scope, sycl::memory_scope::device,
                   sycl::memory_scope::system,
                   sycl::memory_scope::work_group>::generate_named();
    const auto address_spaces = value_pack<
        sycl::access::address_space, sycl::access::address_space::local_space,
        sycl::access::address_space::generic_space>::generate_named();

    for_all_combinations<atomicity_work_group_scope, T>(
        memory_orders, memory_scopes, address_spaces, type_name);
  }
};

template <typename T>
struct run_aquire_release {
  void operator()(const std::string& type_name) {
    const auto memory_orders =
        value_pack<sycl::memory_order, sycl::memory_order::acq_rel,
                   sycl::memory_order::seq_cst>::generate_named();
    const auto memory_scopes =
        value_pack<sycl::memory_scope, sycl::memory_scope::device,
                   sycl::memory_scope::system,
                   sycl::memory_scope::work_group>::generate_named();
    const auto address_spaces = value_pack<
        sycl::access::address_space, sycl::access::address_space::local_space,
        sycl::access::address_space::generic_space>::generate_named();

    for_all_combinations<aquire_release, T>(memory_orders, memory_scopes,
                                            address_spaces, type_name);
  }
};

template <typename T>
struct run_ordering {
  void operator()(const std::string& type_name) {
    const auto memory_orders =
        value_pack<sycl::memory_order, sycl::memory_order::release,
                   sycl::memory_order::seq_cst>::generate_named();
    const auto memory_scopes =
        value_pack<sycl::memory_scope, sycl::memory_scope::device,
                   sycl::memory_scope::system,
                   sycl::memory_scope::work_group>::generate_named();
    const auto address_spaces = value_pack<
        sycl::access::address_space, sycl::access::address_space::local_space,
        sycl::access::address_space::generic_space>::generate_named();

    for_all_combinations<ordering, T>(memory_orders, memory_scopes,
                                      address_spaces, type_name);
  }
};
#ifdef __cpp_lib_atomic_ref
template <typename T>
struct run_atomicity_with_host_code {
  void operator()(const std::string& type_name) {
    const auto memory_orders =
        value_pack<sycl::memory_order, sycl::memory_order::relaxed,
                   sycl::memory_order::acq_rel,
                   sycl::memory_order::seq_cst>::generate_named();
    const auto memory_scopes =
        value_pack<sycl::memory_scope,
                   sycl::memory_scope::system>::generate_named();
    const auto address_spaces = value_pack<
        sycl::access::address_space, sycl::access::address_space::global_space,
        sycl::access::address_space::generic_space>::generate_named();

    for_all_combinations<atomicity_with_host_code, T>(
        memory_orders, memory_scopes, address_spaces, type_name);
  }
};
#endif
}  // namespace atomic_ref_stress_test
#endif  // SYCL_CTS_ATOMIC_REF_STRESS_TEST_H
