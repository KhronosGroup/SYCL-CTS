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

namespace group_interface::tests {

template <int Dimensions>
static bool testWorkGroup(sycl::nd_item<Dimensions> it) {
  bool passed = true;
  sycl::khr::work_group<Dimensions> work_group{it.get_group()};
  sycl::group<Dimensions> group{it.get_group()};

  // id
  static_assert(
      std::is_same_v<decltype(work_group.id()),
                     typename sycl::khr::work_group<Dimensions>::id_type>);
  passed &= (group.get_group_id() == work_group.id());

  // linear_id
  static_assert(std::is_same_v<
                decltype(work_group.linear_id()),
                typename sycl::khr::work_group<Dimensions>::linear_id_type>);
  passed &= (group.get_group_linear_id() == work_group.linear_id());

  // range
  static_assert(
      std::is_same_v<decltype(work_group.range()),
                     typename sycl::khr::work_group<Dimensions>::range_type>);
  passed &= (group.get_group_range() == work_group.range());

  // size
  static_assert(
      std::is_same_v<decltype(work_group.size()),
                     typename sycl::khr::work_group<Dimensions>::size_type>);
  passed &= (group.get_local_linear_range() == work_group.size());

  // leader_of
  static_assert(
      std::is_same_v<decltype(sycl::khr::leader_of(work_group)), bool>);
  passed &= (group.leader() == sycl::khr::leader_of(work_group));

  return passed;
}

template <int Dimensions>
static bool testSubGroup(sycl::nd_item<Dimensions> it) {
  bool passed = true;
  sycl::khr::sub_group sub_group{it.get_sub_group()};
  sycl::sub_group group{it.get_sub_group()};

  // id
  static_assert(std::is_same_v<decltype(sub_group.id()),
                               typename sycl::khr::sub_group::id_type>);
  passed &= (group.get_group_id() == sub_group.id());

  // linear_id
  static_assert(std::is_same_v<decltype(sub_group.linear_id()),
                               typename sycl::khr::sub_group::linear_id_type>);
  passed &= (group.get_group_linear_id() == sub_group.linear_id());

  // range
  static_assert(std::is_same_v<decltype(sub_group.range()),
                               typename sycl::khr::sub_group::range_type>);
  passed &= (group.get_group_range() == sub_group.range());

  // size
  static_assert(std::is_same_v<decltype(sub_group.size()),
                               typename sycl::khr::sub_group::size_type>);
  passed &= (group.get_local_range()[0] == sub_group.size());

  // max_size
  static_assert(std::is_same_v<decltype(sub_group.max_size()),
                               typename sycl::khr::sub_group::size_type>);
  passed &= (group.get_max_local_range()[0] == sub_group.max_size());

  // leader_of
  static_assert(
      std::is_same_v<decltype(sycl::khr::leader_of(sub_group)), bool>);
  passed &= (group.leader() == sycl::khr::leader_of(sub_group));

  return passed;
}

template <int Dimensions>
static bool testWorkItemGroup(sycl::nd_item<Dimensions> it) {
  bool passed = true;
  sycl::group<Dimensions> group{it.get_group()};
  sycl::khr::work_group<Dimensions> work_group{group};
  sycl::khr::work_item item{sycl::khr::get_item(work_group)};

  // id
  static_assert(
      std::is_same_v<decltype(item.id()),
                     typename sycl::khr::work_item<
                         sycl::khr::work_group<Dimensions>>::id_type>);
  passed &= (group.get_local_id() == item.id());

  // linear_id
  static_assert(
      std::is_same_v<decltype(item.linear_id()),
                     typename sycl::khr::work_item<
                         sycl::khr::work_group<Dimensions>>::linear_id_type>);
  passed &= (group.get_local_linear_id() == item.linear_id());

  // range
  static_assert(
      std::is_same_v<decltype(item.range()),
                     typename sycl::khr::work_item<
                         sycl::khr::work_group<Dimensions>>::range_type>);
  passed &= (group.get_local_range() == item.range());

  // size
  static_assert(
      std::is_same_v<decltype(item.size()),
                     typename sycl::khr::work_item<
                         sycl::khr::work_group<Dimensions>>::size_type>);
  passed &= (1 == item.size());

  return passed;
}

template <int Dimensions>
static bool testWorkItemSubgroup(sycl::nd_item<Dimensions> it) {
  bool passed = true;
  sycl::sub_group group{it.get_sub_group()};
  sycl::khr::sub_group sub_group{group};
  sycl::khr::work_item item{sycl::khr::get_item(sub_group)};

  // id
  static_assert(std::is_same_v<
                decltype(item.id()),
                typename sycl::khr::work_item<sycl::khr::sub_group>::id_type>);
  passed &= (group.get_local_id() == item.id());

  // linear_id
  static_assert(
      std::is_same_v<
          decltype(item.linear_id()),
          typename sycl::khr::work_item<sycl::khr::sub_group>::linear_id_type>);
  passed &= (group.get_local_linear_id() == item.linear_id());

  // range
  static_assert(
      std::is_same_v<
          decltype(item.range()),
          typename sycl::khr::work_item<sycl::khr::sub_group>::range_type>);
  passed &= (group.get_local_range() == item.range());

  // size
  static_assert(
      std::is_same_v<
          decltype(item.size()),
          typename sycl::khr::work_item<sycl::khr::sub_group>::size_type>);
  passed &= (1 == item.size());

  return passed;
}

template <int Dimensions>
static void testGroupInterface(sycl::nd_range<Dimensions> nd_range) {
  constexpr std::size_t TEST_COUNT = 4;
  sycl::queue q{sycl_cts::util::get_cts_object::queue()};
  sycl::buffer<bool> results{nd_range.get_global_range().size() * TEST_COUNT};
  results.get_range();
  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      acc[(it.get_global_linear_id() * TEST_COUNT) + 0] = testWorkGroup(it);
      acc[(it.get_global_linear_id() * TEST_COUNT) + 1] = testSubGroup(it);
      acc[(it.get_global_linear_id() * TEST_COUNT) + 2] = testWorkItemGroup(it);
      acc[(it.get_global_linear_id() * TEST_COUNT) + 3] =
          testWorkItemSubgroup(it);
    });
  });
  sycl::host_accessor acc{results, sycl::read_only};
  for (auto result : acc) CHECK(result);
}

TEST_CASE(
    "the group interface extension defines the SYCL_KHR_GROUP_INTERFACE "
    "macro",
    "[khr_group_interface]") {
#ifndef SYCL_KHR_GROUP_INTERFACE
  static_assert(false, "SYCL_KHR_GROUP_INTERFACE is not defined");
#endif
}

TEST_CASE(
    "the group interface extension correctly defines the work_group, "
    "sub_group, and work_item classes",
    "[khr_group_interface]") {
  testGroupInterface(sycl::nd_range<1>(100, 10));
  testGroupInterface(sycl::nd_range<2>({10, 12}, {2, 3}));
  testGroupInterface(sycl::nd_range<3>({3, 10, 12}, {1, 2, 3}));
}

}  // namespace group_interface::tests
