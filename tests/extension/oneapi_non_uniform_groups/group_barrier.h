/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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
*******************************************************************************/

#include "../../group_functions/group_functions_common.h"
#include "non_uniform_group_common.h"

template <typename GroupT>
class non_uniform_group_barrier_kernel;

/**
 * @brief Provides test for arbitraty non-uniform group barriers
 * @tparam GroupT Type of the non-uniform group to test with
 */
template <typename GroupT>
struct non_uniform_group_barrier_test {
  void operator()() {
    auto queue = once_per_unit::get_queue();
    const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

    INFO("Testing group-of predicate function for " + group_name);
    if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
      SKIP("Device does not support " + group_name);
    }

    std::vector<sycl::memory_scope> supported_barriers =
        queue.get_context()
            .get_info<sycl::info::context::atomic_fence_scope_capabilities>();

    using sms = std::tuple<sycl::memory_scope, bool, bool>;
    // indices of the tuple components
    enum s { scope = 0, support = 1, test = 2 };

    constexpr int non_uniform_group_barrier_variants = 5;
    std::array<sms, non_uniform_group_barrier_variants>
        non_uniform_group_barriers{
            {{sycl::memory_scope::sub_group, true, true},
             {sycl::memory_scope::sub_group, true, true},
             {sycl::memory_scope::work_group, true, true},
             {sycl::memory_scope::device, true, true},
             {sycl::memory_scope::system, true, true}}};
    std::array<std::string, non_uniform_group_barrier_variants>
        non_uniform_group_barriers_names{
            "default", "sycl::memory_scope::sub_group",
            "sycl::memory_scope::work_group", "sycl::memory_scope::device",
            "sycl::memory_scope::system"};
    for (auto& barrier : non_uniform_group_barriers) {
      auto& sb = supported_barriers;
      if (std::find(sb.begin(), sb.end(), std::get<s::scope>(barrier)) ==
          sb.end()) {
        std::get<s::support>(barrier) = false;
      }
    }

    using el_type = int32_t;
    sycl::device device = queue.get_device();

    // Check the maximum number elements of type "el_type" that can be
    // placed in the device's global and local memory. Since the test
    // tries to allocate local and global buffers with a size equal to
    // the work group size, the latter must be limited by the allowed
    // buffer size.
    uint64_t global_mem_size_in_bytes =
        device.get_info<sycl::info::device::max_mem_alloc_size>();
    uint64_t global_mem_size_in_elements =
        global_mem_size_in_bytes / sizeof(el_type);

    uint64_t local_mem_size_in_bytes =
        device.get_info<sycl::info::device::local_mem_size>();
    uint64_t local_mem_size_in_elements =
        local_mem_size_in_bytes / sizeof(el_type);

    uint64_t work_items_limit =
        std::min(global_mem_size_in_elements, local_mem_size_in_elements);

    sycl::range<1> work_group_range =
        sycl_cts::util::work_group_range<1>(queue, work_items_limit);
    size_t work_group_size = work_group_range.size();

    for (size_t test_case = 0;
         test_case < NonUniformGroupHelper<GroupT>::num_test_cases;
         ++test_case) {
      const std::string test_case_name =
          NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
      INFO("Running test case (" + std::to_string(test_case) + ") with " +
           test_case_name);

      std::vector<int32_t> v(work_group_size, 0);
      sycl::buffer<int32_t, 1> global_mem(v.data(),
                                          sycl::range<1>(work_group_size));

      sycl::buffer<sms, 1> non_uniform_group_barriers_buf(
          non_uniform_group_barriers.data(),
          sycl::range<1>(non_uniform_group_barrier_variants));

      queue.submit([&](sycl::handler& cgh) {
        sycl::nd_range<1> executionRange(work_group_range, work_group_range);

        auto non_uniform_group_barriers_acc =
            non_uniform_group_barriers_buf
                .get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<int32_t, 1> local_acc(
            sycl::range<1>(work_group_size), cgh);
        sycl::accessor<int32_t, 1> global_acc =
            global_mem.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<non_uniform_group_barrier_kernel<GroupT>>(
            executionRange, [=](sycl::nd_item<1> item) {
              sycl::sub_group sub_group = item.get_sub_group();

              // If this item is not participating in the group, leave early.
              if (!NonUniformGroupHelper<GroupT>::should_participate(sub_group,
                                                                     test_case))
                return;

              GroupT non_uniform_group =
                  NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

              size_t llid = non_uniform_group.get_local_linear_id();
              size_t max_id = non_uniform_group.get_local_linear_range() - 1;

              static_assert(
                  std::is_same_v<void, decltype(sycl::group_barrier(
                                           non_uniform_group))>,
                  "Return type of group_barrier(GroupT g) is wrong\n");
              static_assert(
                  std::is_same_v<void, decltype(sycl::group_barrier(
                                           non_uniform_group,
                                           non_uniform_group.fence_scope))>,
                  "Return type of group_barrier(GroupT g, "
                  "memory_scope fence_scope) is wrong\n");

              // test of default barrier
              local_acc[llid] = llid;
              sycl::group_barrier(non_uniform_group);

              if (local_acc[max_id - llid] != max_id - llid)
                std::get<s::test>(non_uniform_group_barriers_acc[0]) = false;
              sycl::group_barrier(non_uniform_group);

              local_acc[llid] = 1;
              sycl::group_barrier(non_uniform_group);

              if (local_acc[max_id - llid] != 1)
                std::get<s::test>(non_uniform_group_barriers_acc[0]) = false;
              sycl::group_barrier(non_uniform_group);

              // tests for other barriers
              for (int i = 1; i < non_uniform_group_barrier_variants; ++i) {
                auto& barrier = non_uniform_group_barriers_acc[i];

                if ((sub_group.get_group_linear_id() == 0) &&
                    (non_uniform_group.get_group_linear_id() ==
                     NonUniformGroupHelper<GroupT>::
                         preferred_single_worker_group_id(test_case)) &&
                    std::get<s::support>(barrier)) {
                  local_acc[llid] = llid;
                  global_acc[llid] = llid;
                  sycl::group_barrier(non_uniform_group);

                  if (local_acc[max_id - llid] != max_id - llid ||
                      global_acc[max_id - llid] != max_id - llid)
                    std::get<s::test>(barrier) = false;
                  sycl::group_barrier(non_uniform_group);

                  switch (std::get<s::scope>(barrier)) {
                    case sycl::memory_scope::sub_group:
                    case sycl::memory_scope::work_group:
                      local_acc[llid] = 1;
                      sycl::group_barrier(non_uniform_group,
                                          std::get<s::scope>(barrier));

                      if (local_acc[max_id - llid] != 1)
                        std::get<s::test>(barrier) = false;
                      sycl::group_barrier(non_uniform_group);

                      [[fallthrough]];
                    default:
                      global_acc[llid] = 1;
                      sycl::group_barrier(non_uniform_group,
                                          std::get<s::scope>(barrier));

                      if (global_acc[max_id - llid] != 1)
                        std::get<s::test>(barrier) = false;
                      sycl::group_barrier(non_uniform_group);
                  }
                }
              }
            });
      });

      for (int i = 0; i < non_uniform_group_barrier_variants; ++i) {
        bool result = std::get<s::test>(non_uniform_group_barriers[i]);
        std::string work_group =
            sycl_cts::util::work_group_print(work_group_range);
        CAPTURE(group_name, work_group);
        INFO("Result of group_barrier invocation for sub-group and "
             << non_uniform_group_barriers_names[i] << " memory scope is "
             << (result ? "right" : "wrong"));
        CHECK(result);
      }
    }
  }
};
