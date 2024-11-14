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

#include "../../common/common.h"

namespace root_group::tests {

#ifdef SYCL_EXT_ONEAPI_ROOT_GROUP

template <typename KernelName, bool UseRootSync, size_t... Dims>
static void check_root_group_api() {
  constexpr int Dimensions = sizeof...(Dims);
  auto q = sycl_cts::util::get_cts_object::queue();
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  auto kernel = bundle.get_kernel<KernelName>();
  auto local_range = sycl::range<Dimensions>(Dims...);
  auto maxWGs = kernel.template ext_oneapi_get_info<
      sycl::ext::oneapi::experimental::info::kernel_queue_specific::
          max_num_work_groups>(q, local_range, 0);
  REQUIRE(maxWGs >= 1);
  auto global_range = local_range;
  global_range[0] *= maxWGs;
  REQUIRE(global_range.size() == local_range.size() * maxWGs);
  auto nd_range = sycl::nd_range<Dimensions>{global_range, local_range};
  auto props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};

  struct checks {
    enum {
      get_group_id,
      get_local_id,
      get_group_range,
      get_local_range,
      get_max_local_range,
      get_group_linear_id,
      get_local_linear_id,
      get_group_linear_range,
      get_local_linear_range,
      leader,
      COUNT,
    };
  };

  using CheckResults = bool[checks::COUNT];
  sycl::buffer<CheckResults, Dimensions> results_buffer{global_range};

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results_buffer, cgh, sycl::write_only};
    const auto kernel = [=](auto it) -> void {
      auto& results = acc[it.get_global_id()];

      auto root = it.ext_oneapi_get_root_group();
      results[checks::get_group_id] =
          root.get_group_id() == sycl::range<Dimensions>();
      results[checks::get_local_id] = root.get_local_id() == it.get_global_id();
      results[checks::get_group_range] =
          root.get_group_range() == (1 | sycl::range<Dimensions>());
      results[checks::get_local_range] =
          root.get_local_range() == it.get_global_range();
      results[checks::get_max_local_range] =
          root.get_max_local_range() == root.get_local_range();
      results[checks::get_group_linear_id] = root.get_group_linear_id() == 0;
      results[checks::get_local_linear_id] =
          root.get_local_linear_id() == it.get_global_linear_id();
      results[checks::get_group_linear_range] =
          root.get_group_linear_range() == root.get_group_range().size();
      results[checks::get_local_linear_range] =
          root.get_local_linear_range() == root.get_local_range().size();
      results[checks::leader] = root.leader() == (root.get_local_id() == 0);
    };
    if constexpr (UseRootSync) {
      cgh.parallel_for<KernelName>(nd_range, props, kernel);
    } else {
      cgh.parallel_for<KernelName>(nd_range, kernel);
    }
  });
  q.wait();

  CheckResults results = {};
  sycl::accessor acc = results_buffer.get_host_access();
  for (int check = 0; check < checks::COUNT; check++) {
    bool passed = true;
    for (const auto& it : acc) {
      passed &= it[check];
    }
    results[check] = passed;
  }

  using rg = sycl::ext::oneapi::experimental::root_group<Dimensions>;

  // type aliases
  CHECK(std::is_same_v<typename rg::id_type, sycl::id<Dimensions>>);
  CHECK(std::is_same_v<typename rg::range_type, sycl::range<Dimensions>>);
  CHECK(std::is_same_v<typename rg::linear_id_type, size_t>);

  // static members
  CHECK(std::is_same_v<std::remove_const_t<decltype(rg::dimensions)>, int>);
  CHECK(rg::dimensions == Dimensions);
  CHECK(std::is_same_v<std::remove_const_t<decltype(rg::fence_scope)>,
                       sycl::memory_scope>);
  CHECK(rg::fence_scope == sycl::memory_scope::device);

  // get_group_id
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_group_id()),
                       sycl::id<Dimensions>>);
  CHECK(results[checks::get_group_id]);

  // get_local_id
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_local_id()),
                       sycl::id<Dimensions>>);
  CHECK(results[checks::get_local_id]);

  // get_group_range
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_group_range()),
                       sycl::range<Dimensions>>);
  CHECK(results[checks::get_group_range]);

  // get_local_range
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_local_range()),
                       sycl::range<Dimensions>>);
  CHECK(results[checks::get_local_range]);

  // get_max_local_range
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_max_local_range()),
                       sycl::range<Dimensions>>);
  CHECK(results[checks::get_max_local_range]);

  // get_group_linear_id
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_group_linear_id()),
                       size_t>);
  CHECK(results[checks::get_group_linear_id]);

  // get_local_linear_id
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_local_linear_id()),
                       size_t>);
  CHECK(results[checks::get_local_linear_id]);

  // get_group_linear_range
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_group_linear_range()),
                       size_t>);
  CHECK(results[checks::get_group_linear_range]);

  // get_local_linear_range
  CHECK(std::is_same_v<decltype(std::declval<rg>().get_local_linear_range()),
                       size_t>);
  CHECK(results[checks::get_local_linear_range]);

  // leader
  CHECK(std::is_same_v<decltype(std::declval<rg>().leader()), bool>);
  CHECK(results[checks::leader]);
}

template <typename KernelName, size_t... Dims>
static void check_root_group_barrier() {
  constexpr int Dimensions = sizeof...(Dims);
  auto q = sycl_cts::util::get_cts_object::queue();
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  auto kernel = bundle.get_kernel<KernelName>();
  auto local_range = sycl::range<Dimensions>(Dims...);
  auto maxWGs = kernel.template ext_oneapi_get_info<
      sycl::ext::oneapi::experimental::info::kernel_queue_specific::
          max_num_work_groups>(q, local_range, 3);
  REQUIRE(maxWGs >= 1);
  auto global_range = local_range;
  global_range[0] *= maxWGs;
  REQUIRE(global_range.size() == local_range.size() * maxWGs);
  auto nd_range = sycl::nd_range<Dimensions>{global_range, local_range};
  auto props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};

  const int N = static_cast<int>(global_range.size());
  int* data = sycl::malloc_shared<int>(N, q);
  std::fill_n(data, N, 0);

  q.parallel_for<KernelName>(nd_range, props, [=](auto it) {
    auto root = it.ext_oneapi_get_root_group();
    data[root.get_local_linear_id()] = 1;
    sycl::group_barrier(root);
    int sum = 0;
    for (int i = N - 1; i >= 0; i--) {
      sum += data[i];
    }
    sycl::group_barrier(root);
    data[root.get_local_linear_id()] = (sum == N ? 1 : 0);
  });
  q.wait();

  for (int i = 0; i < N; i++) {
    CHECK(data[i] == 1);
  }
  sycl::free(data, q);
}

#endif

TEST_CASE(
    "Test for \"Root Group\" extension, check root_group class and "
    "get_child_group API functionality",
    "[oneapi_root_group_api]") {
#ifndef SYCL_EXT_ONEAPI_ROOT_GROUP
  SKIP("SYCL_EXT_ONEAPI_ROOT_GROUP is not defined");
#else
  check_root_group_api<struct RootGroupNoSyncProp1D, false, 4>();
  check_root_group_api<struct RootGroupNoSyncProp2D, false, 6, 4>();
  check_root_group_api<struct RootGroupNoSyncProp3D, false, 8, 6, 4>();
  check_root_group_api<struct RootGroupSyncProp1D, true, 4>();
  check_root_group_api<struct RootGroupSyncProp2D, true, 6, 4>();
  check_root_group_api<struct RootGroupSyncProp3D, true, 8, 6, 4>();
#endif
}

TEST_CASE(
    "Test for \"Root Group\" extension, check root_group synchronization using "
    "group_barrier function"
    "[oneapi_root_group_barrier]") {
#ifndef SYCL_EXT_ONEAPI_ROOT_GROUP
  SKIP("SYCL_EXT_ONEAPI_ROOT_GROUP is not defined");
#else
  check_root_group_barrier<struct RootGroupBarrier1D, 4>();
  check_root_group_barrier<struct RootGroupBarrier2D, 6, 4>();
  check_root_group_barrier<struct RootGroupBarrier3D, 8, 6, 4>();
#endif
}

}  // namespace root_group::tests
