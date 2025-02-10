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
//  Provides sycl::atomic_fence function test
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../common/section_name_builder.h"
#include "../common/type_coverage.h"

namespace atomic_fence_test {

// FIXME: re-enable when support for atomic_fence is implemented in AdaptiveCpp
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

constexpr int expected_val = 42;

// enum covers two types of check:
enum class test_type {
  single_group = 1,    // between items in the single group
  between_groups = 2,  // between items in multiple groups
};

/**
 * @brief Function for checking device atomic_fence_order capability
 * @param queue SYCL queue is used to access device parameters
 * @param order memory_order value used to check
 * @param order_name String for name of the order value
 * @return true if device supports atomic_fence with memory_order order
 */
bool check_atomic_fence_order_capability(sycl::queue& queue,
                                         sycl::memory_order order,
                                         const std::string& order_name) {
  auto orders =
      queue.get_device()
          .get_info<sycl::info::device::atomic_fence_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), order) == orders.end()) {
    WARN(std::string("Device does not support atomic_fence with order = ") +
         order_name);
    return false;
  }
  return true;
}

/**
 * @brief Function for checking device atomic_fence_scope capability
 * @param queue SYCL queue is used to access device parameters
 * @param scope memory_scope value used to check
 * @param scope_name String for name of the scope value
 * @return true if device supports atomic_fence with memory_scope scope
 */
bool check_atomic_fence_scope_capability(sycl::queue& queue,
                                         sycl::memory_scope scope,
                                         const std::string& scope_name) {
  auto scopes =
      queue.get_device()
          .get_info<sycl::info::device::atomic_fence_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), scope) == scopes.end()) {
    WARN(std::string("Device does not support atomic_fence with scope = ") +
         scope_name);
    return false;
  }
  return true;
}

/**
 * @brief Function for checking device atomic_fence_scope and atomic_fence_order
 * capabilities
 * @param queue queue SYCL queue is used to access device parameters
 * @param order memory_order value used to check
 * @param scope memory_scope value used to check
 * @param order_name String for name of the order value
 * @param scope_name String for name of the scope value
 * @return true if device supports atomic_fence with memory_scope scope and
 * memory_order order
 */
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

/**
 * @brief Common functor to check sycl::atomic_fence
 * This check algorithm checks that using a synchronizing variable of
 * atomic_ref type in combination with atomic_fence function provides no data
 * racing and the specified order of instruction execution.
 * Also, in the algorithm the atomic_fence function check will be performed
 * only if the leader writes a value to the sync_flag variable before all other
 * items complete the loop with the check of sync_flag value.
 * It does not give guarantee that the check will be performed,
 * but while there is no guarantee that the loop with the condition
 * while (sync_flag != true); will always complete, a safe and less strict check
 * algorithm has been chosen.
 *
 * @tparam OrderT memory_order value
 * @tparam ScopeT memory_scope value
 * @tparam TestT test_type value
 */
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
      auto queue = once_per_unit::get_queue();
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

      // Count of retries in the check cycle
      constexpr size_t RETRY_COUNT = 256;

      bool res = true;
      int sync = 0;
      int data = 0;
      int value = expected_val;

      // These global_range and local_range values provide a check in one group
      // when test_type = single_group, and between four groups when
      // test_type = between_groups
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
          cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                           [=](sycl::nd_item<1> nditem) {
                             sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                              sycl::memory_scope::work_group>
                                 sync_flag(sync_flag_acc[0]);
                             int* data = &data_acc[0];
                             // Only one nditem should perform non-atomic write.
                             // All other nditems should perform non-atomic
                             // reads
                             if (is_specified_item_in_kernel(nditem)) {
                               // Non-atomic write to data
                               *data = value;
                               // Used atomic_fence to guarantee the order
                               // instructions execution
                               sycl::atomic_fence(order_write, MemoryScope);
                               // Used atomic sync flag to avoid data raicing
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
                               // After the fence safe non-atomic reading
                               if (write_happened) {
                                 // Non-atomic read of data
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
  /**
   * @brief The function checks that nditem exactly is the one that writes
   * value to a variable
   *
   * @return true if nditem writes value
   */
  static bool is_specified_item_in_kernel(const sycl::nd_item<1>& nditem) {
    auto g = nditem.get_group();
    if (g.leader() && test_type::single_group == TestType) {
      return true;
    } else if (0 == nditem.get_global_linear_id() &&
               test_type::between_groups == TestType) {
      return true;
    }
    return false;
  }

  /**
   * @brief Function to get accessor of proper type depends on the TestType
   * value
   */
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
#endif  // !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

// FIXME: re-enable when support for atomic_fence is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_fence function",
 "[atomic_fence]")({ atomic_fence_test::run_test{}(); });

}  // namespace atomic_fence_test
