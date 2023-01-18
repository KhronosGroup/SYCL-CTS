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

// FIXME: re-enable when support for unnamed kernels is implemented in
// ComputeCpp and atomic_fence is implemented in hipSYCL
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

namespace atomic_fence_test {

constexpr int expected_val = 42;

enum class test_type {
  single_group = 1,
  between_groups = 2,
};

bool check_atomic_fence_order_capability(sycl::queue& queue,
                                         sycl::memory_order order,
                                         const std::string& order_name) {
// FIXME: re-enable when sycl::info::device::atomic_fence_order_capabilities is
// implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
  / auto orders =
      queue.get_device()
          .get_info<sycl::info::device::atomic_fence_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), order) == orders.end()) {
    WARN(std::string("Device does not support atomic_fence with order = ") +
         order_name);
    return false;
  }
#endif
  return true;
}

bool check_atomic_fence_scope_capability(sycl::queue& queue,
                                         sycl::memory_scope scope,
                                         const std::string& scope_name) {
// FIXME: re-enable when sycl::info::device::atomic_fence_scope_capabilities is
// implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
  auto scopes =
      queue.get_device()
          .get_info<sycl::info::device::atomic_fence_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), scope) == orders.end()) {
    WARN(std::string("Device does not support atomic_fence with scope = ") +
         scope_name);
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
 * @brief Factory function for getting type_pack with test_type values for
 * atomic_fence
 */
inline auto get_test_types() {
  static const auto test_types =
      value_pack<test_type, test_type::single_group,
                 test_type::between_groups>::generate_named();
  return test_types;
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

bool check_memory_order_scope_capabilities(sycl::queue& queue,
                                           sycl::memory_order order,
                                           sycl::memory_scope scope,
                                           const std::string& order_name,
                                           const std::string& scope_name) {
  if (!check_atomic_fence_order_capability(queue, order, order_name)) {
    return false;
  }
  if (sycl::memory_order::release == order) {
    if (!check_atomic_fence_order_capability(queue, sycl::memory_order::acquire,
                                             "memory_order::acquire")) {
      return false;
    }
  }

  if (!check_atomic_fence_scope_capability(queue, scope, scope_name)) {
    return false;
  }
  return true;
}

template <typename OrderT, typename ScopeT, typename TestT>
class run_atomic_fence {
  static constexpr sycl::memory_order MemoryOrder = OrderT::value;
  static constexpr sycl::memory_scope MemoryScope = ScopeT::value;
  static constexpr test_type TestType = TestT::value;

 public:
  void operator()(const std::string& memory_order_name,
                  const std::string& memory_scope_name,
                  const std::string& test_type_name) {
    SECTION(sycl_cts::section_name(std::string("Check atomic_fence with "
                                               "memory_order = ") +
                                   memory_order_name +
                                   " and scope = " + memory_scope_name +
                                   " and test_type = " + test_type_name)
                .create()) {
      auto queue = sycl_cts::util::get_cts_object::queue();
      if (!check_memory_order_scope_capabilities(queue, MemoryOrder,
                                                 MemoryScope, memory_order_name,
                                                 memory_scope_name)) {
        return;
      }
      sycl::memory_order order_write = MemoryOrder;
      sycl::memory_order order_read = MemoryOrder;
      if (sycl::memory_order::release == order_read) {
        order_read = sycl::memory_order::acquire;
      }

      constexpr size_t RETRY_COUNT = 256;
      bool res = true;
      int sync = 0;
      int data = 0;
      int value = expected_val;
      sycl::range<1> global_range(2);
      if (test_type::between_groups == TestType) {
        global_range = sycl::range<1>(8);
      }
      sycl::range<1> local_range(2);
      {
        sycl::buffer<bool> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<int> sync_buffer(&sync, sycl::range<1>(1));
        sycl::buffer<int> data_buffer(&data, sycl::range<1>(1));
        queue.submit([&](sycl::handler& cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto sync_flag_acc = get_accessor(cgh, sync_buffer);
          auto data_acc = get_accessor(cgh, data_buffer);
          cgh.parallel_for(
              sycl::nd_range<1>(global_range, local_range),
              [=](sycl::nd_item<1> nditem) {
                auto g = nditem.get_group();
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group>
                    sync_flag(sync_flag_acc[0]);
                int* data = &data_acc[0];
                if ((test_type::single_group == TestType && g.leader()) ||
                    (test_type::between_groups == TestType &&
                     0 == nditem.get_global_linear_id())) {
                  *data = value;
                  sycl::atomic_fence(order_write, MemoryScope);
                  sync_flag = 1;
                } else {
                  bool write_happened = false;
                  for (size_t i = 0; i < RETRY_COUNT; i++) {
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

 private:
  auto get_accessor(sycl::handler& cgh, sycl::buffer<int>& buf) {
    if constexpr (test_type::single_group == TestType) {
      return sycl::local_accessor<int>(sycl::range<1>(1), cgh);
    } else {
      return buf.template get_access<sycl::access_mode::read_write>(cgh);
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
    for_all_combinations<run_atomic_fence>(
        memory_orders, memory_scopes_single_group,
        value_pack<test_type, test_type::single_group>::generate_named());

    for_all_combinations<run_atomic_fence>(
        memory_orders, memory_scopes_between_groups,
        value_pack<test_type, test_type::between_groups>::generate_named());
  }
};
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP

// FIXME: re-enable when support for unnamed kernels is implemented in
// ComputeCpp and atomic_fence is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::atomic_fence function",
 "[atomic_fence]")({ atomic_fence_test::run_test{}(); });

}  // namespace atomic_fence_test
