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
*******************************************************************************/

#include "../common/common.h"
#include <catch2/catch_template_test_macros.hpp>

#include "group_functions_common.h"

template <int D>
class test_fence;

TEMPLATE_TEST_CASE_SIG("Group barriers", "[group_func][dim]", ((int D), D), 1,
                       2, 3) {
  auto queue = sycl_cts::util::get_cts_object::queue();

  // FIXME: hipSYCL and DPCPP have no implemented
  //  atomic_fence_scope_capabilities query
  //  Issue to link https://github.com/intel/llvm/issues/8323
#if !(defined(SYCL_CTS_COMPILING_WITH_HIPSYCL) || \
      defined(SYCL_CTS_COMPILING_WITH_DPCPP))
  std::vector<sycl::memory_scope> supported_barriers =
      queue.get_context()
          .get_info<sycl::info::context::atomic_fence_scope_capabilities>();
#else
  // mock for not available info
  std::vector<sycl::memory_scope> supported_barriers{
      sycl::memory_scope::sub_group, sycl::memory_scope::work_group,
      sycl::memory_scope::device, sycl::memory_scope::system};
  WARN(
      "hipSYCL and DPCPP have no implementation of "
      "atomic_fence_scope_capabilities query, suppose all barrier types as "
      "valid.");
#endif

  using sms = std::tuple<sycl::memory_scope, bool, bool>;
  // indices of the tuple components
  enum s { scope = 0, support = 1, test = 2 };

  constexpr int group_barrier_variants = 4;
  std::array<sms, group_barrier_variants> group_barriers{
      {{sycl::memory_scope::work_group, true, true},
       {sycl::memory_scope::work_group, true, true},
       {sycl::memory_scope::device, true, true},
       {sycl::memory_scope::system, true, true}}};
  std::array<std::string, group_barrier_variants> group_barriers_names{
      "default", "sycl::memory_scope::work_group", "sycl::memory_scope::device",
      "sycl::memory_scope::system"};
  for (auto& barrier : group_barriers) {
    auto& sb = supported_barriers;
    if (std::find(sb.begin(), sb.end(), std::get<s::scope>(barrier)) ==
        sb.end()) {
      std::get<s::support>(barrier) = false;
    }
  }

  constexpr int sub_group_barrier_variants = 5;
  std::array<sms, sub_group_barrier_variants> sub_group_barriers{
      {{sycl::memory_scope::sub_group, true, true},
       {sycl::memory_scope::sub_group, true, true},
       {sycl::memory_scope::work_group, true, true},
       {sycl::memory_scope::device, true, true},
       {sycl::memory_scope::system, true, true}}};
  std::array<std::string, sub_group_barrier_variants> sub_group_barriers_names{
      "default", "sycl::memory_scope::sub_group",
      "sycl::memory_scope::work_group", "sycl::memory_scope::device",
      "sycl::memory_scope::system"};
  for (auto& barrier : sub_group_barriers) {
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

  sycl::range<D> work_group_range =
      sycl_cts::util::work_group_range<D>(queue, work_items_limit);
  size_t work_group_size = work_group_range.size();

  std::vector<int32_t> v(work_group_size, 0);
  sycl::buffer<int32_t, 1> global_mem(v.data(),
                                      sycl::range<1>(work_group_size));

  sycl::buffer<sms, 1> group_barriers_buf(group_barriers.data(),
                                          sycl::range<1>(4));
  sycl::buffer<sms, 1> sub_group_barriers_buf(sub_group_barriers.data(),
                                              sycl::range<1>(5));

  queue.submit([&](sycl::handler& cgh) {
    sycl::nd_range<D> executionRange(work_group_range, work_group_range);

    auto group_barriers_acc =
        group_barriers_buf.get_access<sycl::access::mode::read_write>(cgh);
    auto sub_group_barriers_acc =
        sub_group_barriers_buf.get_access<sycl::access::mode::read_write>(cgh);

    sycl::local_accessor<int32_t, 1> local_acc(sycl::range<1>(work_group_size),
                                               cgh);
    sycl::accessor<int32_t, 1> global_acc =
        global_mem.get_access<sycl::access::mode::read_write>(cgh);

    cgh.parallel_for<test_fence<D>>(executionRange, [=](sycl::nd_item<D> item) {
      sycl::group<D> group = item.get_group();
      size_t llid = group.get_local_linear_id();
      size_t max_id = group.get_local_linear_range() - 1;

      ASSERT_RETURN_TYPE(void, sycl::group_barrier(group),
                         "Return type of group_barrier(group g) is wrong\n");
      ASSERT_RETURN_TYPE(void, sycl::group_barrier(group, group.fence_scope),
                         "Return type of group_barrier(group g, memory_scope "
                         "fence_scope) is wrong\n");

      // test of default barrier
      local_acc[llid] = 0;
      sycl::group_barrier(group);

      local_acc[llid] = 1;
      sycl::group_barrier(group);

      if (local_acc[max_id - llid] != 1)
        std::get<s::test>(group_barriers_acc[0]) = false;

      // tests for other barriers
      for (int i = 1; i < group_barrier_variants; ++i) {
        auto& barrier = group_barriers_acc[i];

        if (std::get<s::support>(barrier)) {
          local_acc[llid] = 0;
          global_acc[llid] = 0;

          sycl::group_barrier(group);

          switch (std::get<s::scope>(barrier)) {
            case sycl::memory_scope::work_group:
              local_acc[llid] = 1;
              sycl::group_barrier(group, std::get<s::scope>(barrier));

              if (local_acc[max_id - llid] != 1)
                std::get<s::test>(barrier) = false;

              [[fallthrough]];
            default:
              global_acc[llid] = 1;
              sycl::group_barrier(group, std::get<s::scope>(barrier));

              if (global_acc[max_id - llid] != 1)
                std::get<s::test>(barrier) = false;
          }
        }
      }

      sycl::sub_group sub_group = item.get_sub_group();
      llid = sub_group.get_local_linear_id();
      max_id = sub_group.get_local_linear_range() - 1;

      ASSERT_RETURN_TYPE(
          void, sycl::group_barrier(sub_group),
          "Return type of group_barrier(sub_group g) is wrong\n");
      ASSERT_RETURN_TYPE(void,
                         sycl::group_barrier(group, sub_group.fence_scope),
                         "Return type of group_barrier(sub_group g, "
                         "memory_scope fence_scope) is wrong\n");

      // test of default barrier
      local_acc[llid] = 0;
      sycl::group_barrier(sub_group);

      local_acc[llid] = 1;
      sycl::group_barrier(sub_group);

      if (local_acc[max_id - llid] != 1)
        std::get<s::test>(sub_group_barriers_acc[0]) = false;

      // tests for other barriers
      for (int i = 1; i < sub_group_barrier_variants; ++i) {
        auto& barrier = sub_group_barriers_acc[i];

        if ((sub_group.get_group_linear_id() == 0) &&
            std::get<s::support>(barrier)) {
          local_acc[llid] = 0;
          global_acc[llid] = 0;

          sycl::group_barrier(sub_group);

          switch (std::get<s::scope>(barrier)) {
            case sycl::memory_scope::sub_group:
            case sycl::memory_scope::work_group:
              local_acc[llid] = 1;
              sycl::group_barrier(sub_group, std::get<s::scope>(barrier));

              if (local_acc[max_id - llid] != 1)
                std::get<s::test>(barrier) = false;

              [[fallthrough]];
            default:
              global_acc[llid] = 1;
              sycl::group_barrier(sub_group, std::get<s::scope>(barrier));

              if (global_acc[max_id - llid] != 1)
                std::get<s::test>(barrier) = false;
          }
        }
      }
    });
  });

  for (int i = 0; i < group_barrier_variants; ++i) {
    bool result = std::get<s::test>(group_barriers[i]);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Result of group_barrier invocation for group and "
         << group_barriers_names[i] << " memory scope is "
         << (result ? "right" : "wrong"));
    CHECK(result);
  }
  for (int i = 0; i < sub_group_barrier_variants; ++i) {
    bool result = std::get<s::test>(sub_group_barriers[i]);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Result of group_barrier invocation for sub-group and "
         << sub_group_barriers_names[i] << " memory scope is "
         << (result ? "right" : "wrong"));
    CHECK(result);
  }
}
