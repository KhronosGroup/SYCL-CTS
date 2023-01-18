/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for SYCL atomic_ref constructors
//
*******************************************************************************/
#ifndef SYCL_CTS_ATOMIC_REF_CONSTRUCTORS_H
#define SYCL_CTS_ATOMIC_REF_CONSTRUCTORS_H

#include "atomic_ref_common.h"
#include <type_traits>

namespace atomic_ref_constructors {
using namespace atomic_ref_tests_common;

template <typename T, typename MemoryOrderT, typename MemoryScopeT,
          typename AddressSpaceT>
class run_constructor_tests {
  static constexpr sycl::memory_order MemoryOrder = MemoryOrderT::value;
  static constexpr sycl::memory_scope MemoryScope = MemoryScopeT::value;
  static constexpr sycl::access::address_space AddressSpace =
      AddressSpaceT::value;
  using AtomicRT = sycl::atomic_ref<T, MemoryOrder, MemoryScope, AddressSpace>;
  using ReferencedType = std::remove_pointer_t<T>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &memory_order_name,
                  const std::string &memory_scope_name,
                  const std::string &address_space_name) {
    auto queue = util::get_cts_object::queue();
    if (memory_order_and_scope_are_not_supported(queue, MemoryOrder,
                                                 MemoryScope)) {
      return;
    }
    ReferencedType data = value_operations::init<ReferencedType>(expected_val);
    T value;
    if constexpr (std::is_pointer_v<T>)
      value = &data;
    else
      value = data;
// FIXME: legal address spaces are not yet defined for atomic_ref used on the
// host, it's possible that will be decided that atomic_ref isn't allowed in
// host code at all. It can be tracked in this issue
// https://gitlab.khronos.org/sycl/Specification/-/issues/637. When the decision
// about atomic_ref usage have been done re-enable test running on host side or
// remove it
#if SYCL_CTS_ATOMIC_REF_ON_HOST == 1
    if constexpr (AddressSpace != sycl::access::address_space::local_space) {
      SECTION(get_section_name(type_name, memory_order_name, memory_scope_name,
                               address_space_name,
                               "Check constructors on host")) {
        AtomicRT a_r(value);
        auto result = a_r.load();
        CHECK(result == value);
        STATIC_CHECK(std::is_same_v<decltype(result), T>);

        AtomicRT a_r_copy(a_r);
        auto result_copy = a_r.load();
        CHECK(result_copy == value);
      }
    }
#endif
    SECTION(get_section_name(type_name, memory_order_name, memory_scope_name,
                             address_space_name,
                             "Check constructors on device")) {
      if constexpr (AddressSpace != sycl::access::address_space::global_space) {
        std::array<bool, 2> res = {false, false};
        {
          sycl::buffer result_buf(res.data(), sycl::range(2));

          queue
              .submit([&](sycl::handler &cgh) {
                auto result_accessor =
                    result_buf.template get_access<sycl::access_mode::write>(
                        cgh);
                sycl::local_accessor<T, 1> loc_acc(sycl::range<1>(1), cgh);
                cgh.parallel_for(sycl::nd_range<1>(1, 1),
                                 [=](sycl::nd_item<1>) {
                                   loc_acc[0] = value;
                                   AtomicRT a_r(loc_acc[0]);

                                   auto result = a_r.load();
                                   result_accessor[0] = (result == value);

                                   AtomicRT a_r_copy(a_r);
                                   auto result_copy = a_r.load();
                                   result_accessor[1] = (result_copy == value);
                                 });
              })
              .wait_and_throw();
        }
        {
          INFO("check atomic_ref(T&) (local space)");
          CHECK(res[0]);
        }
        {
          INFO("check copy-constructor (local space)");
          CHECK(res[1]);
        }
      }
      if constexpr (AddressSpace != sycl::access::address_space::local_space) {
        std::array<bool, 2> res = {false, false};
        {
          sycl::buffer data_buf(&value, sycl::range(1));
          sycl::buffer result_buf(res.data(), sycl::range(2));

          queue
              .submit([&](sycl::handler &cgh) {
                auto data_accessor =
                    data_buf.template get_access<sycl::access_mode::read_write>(
                        cgh);
                auto result_accessor =
                    result_buf.template get_access<sycl::access_mode::write>(
                        cgh);
                cgh.single_task([=]() {
                  AtomicRT a_r(data_accessor[0]);

                  auto result = a_r.load();
                  result_accessor[0] = (result == value);

                  AtomicRT a_r_copy(a_r);
                  auto result_copy = a_r.load();
                  result_accessor[1] = (result_copy == value);
                });
              })
              .wait_and_throw();
        }
        {
          INFO("check atomic_ref(T&) (global space)");
          CHECK(res[0]);
        }
        {
          INFO("check copy-constructor (global space)");
          CHECK(res[1]);
        }
      }
    }
  }
};

/**
 * @brief Run tests for sycl::atomic_ref constructor
 */
template <typename T>
struct run_test {
  void operator()(const std::string &type_name) {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes = get_memory_scopes();
    const auto address_spaces = get_address_spaces();

    for_all_combinations<run_constructor_tests, T>(memory_orders, memory_scopes,
                                                   address_spaces, type_name);

    std::string type_name_for_pointer_types = type_name + "*";
    for_all_combinations<run_constructor_tests, T *>(
        memory_orders, memory_scopes, address_spaces,
        type_name_for_pointer_types);
  }
};

}  // namespace atomic_ref_constructors
#endif  // SYCL_CTS_ATOMIC_REF_CONSTRUCTORS_H
