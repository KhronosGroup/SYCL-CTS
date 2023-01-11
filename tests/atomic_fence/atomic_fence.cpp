/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_fence function test
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "../common/common.h"
#include "../common/section_name_builder.h"
#include "../common/type_coverage.h"

namespace atomic_fence {

constexpr int expected_val = 42;

bool check_atomic_fence_order_capability(sycl::queue& queue,
                                         sycl::memory_order order) {
// FIXME: re-enable when sycl::info::device::atomic_fence_order_capabilities is
// implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
  / auto orders =
      queue.get_device()
          .get_info<sycl::info::device::atomic_fence_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), order) == orders.end()) {
    WARN(
        "Device does not support atomic_fence with memory_order = "
        "order");
    return false;
  }
#endif
  return true;
}

bool check_atomic_fence_scope_capability(sycl::queue& queue,
                                         sycl::memory_scope scope) {
// FIXME: re-enable when sycl::info::device::atomic_fence_scope_capabilities is
// implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
  auto scopes =
      queue.get_device()
          .get_info<sycl::info::device::atomic_fence_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), scope) == orders.end()) {
    WARN(
        "Device does not support atomic_fence with memory_scope = "
        "scope");
    return false;
  }
#endif
  return true;
}

/**
 * @brief Factory function for getting type_pack with memory_order values for
 * atomic_fence
 */
inline auto get_memory_orders() {
  static const auto memory_orders =
      value_pack<sycl::memory_order, sycl::memory_order::release,
                 sycl::memory_order::acq_rel,
                 sycl::memory_order::seq_cst>::generate_named();
  return memory_orders;
}

/**
 * @brief Factory function for getting type_pack with memory_scope for
 * atomic_fence inside single work-group
 */
inline auto get_memory_scopes_single_group() {
  static const auto memory_scopes =
      value_pack<sycl::memory_scope, sycl::memory_scope::work_group,
                 sycl::memory_scope::device,
                 sycl::memory_scope::system>::generate_named();
  return memory_scopes;
}

/**
 * @brief Factory function for getting type_pack with memory_scope for
 * atomic_fence between work-grous
 */
inline auto get_memory_scopes_between_work_groups() {
  static const auto memory_scopes =
      value_pack<sycl::memory_scope, sycl::memory_scope::device,
                 sycl::memory_scope::system>::generate_named();
  return memory_scopes;
}

template <typename OrderT, typename ScopeT>
class run_atomic_ref_in_one_group {
  static constexpr sycl::memory_order MemoryOrder = OrderT::value;
  static constexpr sycl::memory_scope MemoryScope = ScopeT::value;

 public:
  void operator()(const std::string& memory_order_name,
                  const std::string& memory_scope_name) {
    SECTION(sycl_cts::section_name(
                std::string("Check atomic_fence inside single work-group with "
                            "memory_order = ") +
                memory_order_name + " and scope = " + memory_scope_name)
                .create()) {
      auto queue = sycl_cts::util::get_cts_object::queue();
      sycl::memory_order order_write = MemoryOrder;
      if (!check_atomic_fence_order_capability(queue, order_write)) {
        return;
      }
      sycl::memory_order order_read = MemoryOrder;
      if (sycl::memory_order::release == order_read) {
        order_read = sycl::memory_order::acquire;
        if (!check_atomic_fence_order_capability(queue, order_read)) {
          return;
        }
      }

      if (!check_atomic_fence_scope_capability(queue, MemoryScope)) {
        return;
      }
      constexpr int RETRY_COUNT = 256;
      bool res = true;
      int value = value_operations::init<int>(expected_val);
      ;
      sycl::range<1> global_range(2);
      sycl::range<1> local_range(2);
      {
        sycl::buffer<bool> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<int> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler& cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          sycl::local_accessor<int> sync_flag_acc(sycl::range<1>(1), cgh);
          sycl::local_accessor<int> data_acc(sycl::range<1>(1), cgh);
          cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                           [=](sycl::nd_item<1> item) {
                             auto g = item.get_group();
                             sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                              sycl::memory_scope::work_group>
                                 sync_flag(sync_flag_acc[0]);
                             int* data = &data_acc[0];
                             if (g.leader()) {
                               *data = value;
                               sycl::atomic_fence(order_write, MemoryScope);
                               sync_flag = 1;
                             } else {
                               bool write_happened = false;
                               for (int i = 0; i < RETRY_COUNT; i++) {
                                 if (sync_flag == 1) {
                                   write_happened = true;
                                   break;
                                 }
                               }
                               sycl::atomic_fence(order_read, MemoryScope);
                               if (write_happened) {
                                 if (*data != value) res_acc[0] = false;
                               }
                             }
                           });
        });
      }
      CHECK(res);
    }
  }
};

template <typename OrderT, typename ScopeT>
class run_atomic_ref_between_groups {
  static constexpr sycl::memory_order MemoryOrder = OrderT::value;
  static constexpr sycl::memory_scope MemoryScope = ScopeT::value;

 public:
  void operator()(const std::string& memory_order_name,
                  const std::string& memory_scope_name) {
    SECTION(
        sycl_cts::section_name(
            std::string(
                "Check atomic_fence between work-groups with memory_order = ") +
            memory_order_name + " and scope = " + memory_scope_name)
            .create()) {
      auto queue = sycl_cts::util::get_cts_object::queue();
      sycl::memory_order order_write = MemoryOrder;
      if (!check_atomic_fence_order_capability(queue, order_write)) {
        return;
      }
      sycl::memory_order order_read = MemoryOrder;
      if (sycl::memory_order::release == order_read) {
        order_read = sycl::memory_order::acquire;
        if (!check_atomic_fence_order_capability(queue, order_read)) {
          return;
        }
      }

      if (!check_atomic_fence_scope_capability(queue, MemoryScope)) {
        return;
      }
      constexpr int RETRY_COUNT = 256;
      bool res = true;
      int sync = 0;
      int data = 0;
      int value = value_operations::init<int>(expected_val);
      sycl::range<1> global_range(8);
      sycl::range<1> local_range(2);
      {
        sycl::buffer<bool> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<int> val_buffer(&value, sycl::range<1>(1));
        sycl::buffer<int> sync_buffer(&sync, sycl::range<1>(1));
        sycl::buffer<int> data_buffer(&data, sycl::range<1>(1));
        queue.submit([&](sycl::handler& cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto sync_flag_acc =
              sync_buffer.template get_access<sycl::access_mode::read_write>(
                  cgh);
          auto data_acc =
              sync_buffer.template get_access<sycl::access_mode::read_write>(
                  cgh);
          cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                           [=](sycl::nd_item<1> item) {
                             auto g = item.get_group();
                             sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                              sycl::memory_scope::work_group>
                                 sync_flag(sync_flag_acc[0]);
                             int* data = &data_acc[0];
                             if (g.leader()) {
                               *data = value;
                               sycl::atomic_fence(order_write, MemoryScope);
                               sync_flag = 1;
                             } else {
                               bool write_happened = false;
                               for (int i = 0; i < RETRY_COUNT; i++) {
                                 if (sync_flag == 1) {
                                   write_happened = true;
                                   break;
                                 }
                               }
                               sycl::atomic_fence(order_read, MemoryScope);
                               if (write_happened) {
                                 if (*data != value) res_acc[0] = false;
                               }
                             }
                           });
        });
      }
      CHECK(res);
    }
  }
};

/**
 * @brief Run tests for sycl::atomic_fence function
 */
class run_test {
 public:
  void operator()() {
    const auto memory_orders = get_memory_orders();
    const auto memory_scopes_single_group = get_memory_scopes_single_group();
    const auto memory_scopes_between_groups =
        get_memory_scopes_between_work_groups();

    for_all_combinations<run_atomic_ref_in_one_group>(
        memory_orders, memory_scopes_single_group);

    for_all_combinations<run_atomic_ref_between_groups>(
        memory_orders, memory_scopes_between_groups);
  }
};

TEST_CASE("sycl::atomic_fence function", "[atomic_fence]") {
  atomic_fence::run_test{}();
};

}  // namespace atomic_fence
