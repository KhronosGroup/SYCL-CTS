/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
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

namespace work_item_queries::tests {

TEST_CASE(
    "the work item queries extension defines the SYCL_KHR_WORK_ITEM_QUERIES "
    "macro",
    "[khr_work_item_queries]") {
#ifndef SYCL_KHR_WORK_ITEM_QUERIES
  static_assert(false, "SYCL_KHR_WORK_ITEM_QUERIES is not defined");
#endif
}

template <size_t... Dims>
static void check_this_nd_item_api() {
  // Define the kernel ranges.
  constexpr int Dimensions = sizeof...(Dims);
  const sycl::range<Dimensions> local_range{Dims...};
  const sycl::range<Dimensions> global_range = local_range;
  const sycl::nd_range<Dimensions> nd_range{global_range, local_range};
  // Launch an ND-range kernel.
  auto q = sycl_cts::util::get_cts_object::queue();
  sycl::buffer<bool, Dimensions> results{global_range};
  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      // Compare it to this_nd_item<Dimensions>().
      acc[it.get_global_id()] = (it == sycl::khr::this_nd_item<Dimensions>());
    });
  });
  // Check the test results.
  sycl::host_accessor acc{results};
  for (const auto& result : acc) CHECK(result);
}

TEST_CASE("sycl::khr::this_nd_item returns the current nd_item",
          "[khr_work_item_queries]") {
  check_this_nd_item_api<2>();
  check_this_nd_item_api<2, 3>();
  check_this_nd_item_api<2, 3, 4>();
}

template <size_t... Dims>
static void check_this_group_api() {
  // Define the kernel ranges.
  constexpr int Dimensions = sizeof...(Dims);
  const sycl::range<Dimensions> local_range{Dims...};
  const sycl::range<Dimensions> global_range = local_range;
  const sycl::nd_range<Dimensions> nd_range{global_range, local_range};
  // Launch an ND-range kernel.
  auto q = sycl_cts::util::get_cts_object::queue();
  sycl::buffer<bool, Dimensions> results{global_range};
  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      // Compare it.get_group() to this_group<Dimensions>().
      acc[it.get_global_id()] =
          (it.get_group() == sycl::khr::this_group<Dimensions>());
    });
  });
  // Check the test results.
  sycl::host_accessor acc{results};
  for (const auto& result : acc) CHECK(result);
}

TEST_CASE("sycl::khr::this_group returns the current group",
          "[khr_work_item_queries]") {
  check_this_group_api<2>();
  check_this_group_api<2, 3>();
  check_this_group_api<2, 3, 4>();
}

template <size_t... Dims>
static void check_this_sub_group_api() {
  // Define the kernel ranges.
  constexpr int Dimensions = sizeof...(Dims);
  const sycl::range<Dimensions> local_range{Dims...};
  const sycl::range<Dimensions> global_range = local_range;
  const sycl::nd_range<Dimensions> nd_range{global_range, local_range};
  // Launch an ND-range kernel.
  auto q = sycl_cts::util::get_cts_object::queue();
  sycl::buffer<bool, Dimensions> results{global_range};
  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      // Compare it.get_sub_group() to this_sub_group().
      acc[it.get_global_id()] =
          (it.get_sub_group() == sycl::khr::this_sub_group());
    });
  });
  // Check the test results.
  sycl::host_accessor acc{results};
  for (const auto& result : acc) CHECK(result);
}

TEST_CASE("sycl::khr::this_sub_group returns the current sub-group",
          "[khr_work_item_queries]") {
  check_this_sub_group_api<2>();
  check_this_sub_group_api<2, 3>();
  check_this_sub_group_api<2, 3, 4>();
}

}  // namespace work_item_queries::tests
