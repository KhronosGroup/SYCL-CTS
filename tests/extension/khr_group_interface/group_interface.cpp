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

#if __cplusplus >= 202302L
  // extents
  static_assert(
      std::is_same_v<decltype(work_group.extents()),
                     typename sycl::khr::work_group<Dimensions>::extents_type>);
  {
    const sycl::range<Dimensions> localRange = group.get_local_range();
    if constexpr (Dimensions == 1)
      passed &= (work_group.extents() ==
                 std::dextents<std::size_t, Dimensions>(localRange[0]));
    else if constexpr (Dimensions == 2)
      passed &= (work_group.extents() == std::dextents<std::size_t, Dimensions>(
                                             localRange[0], localRange[1]));
    else if constexpr (Dimensions == 3)
      passed &= (work_group.extents() ==
                 std::dextents<std::size_t, Dimensions>(
                     localRange[0], localRange[1], localRange[2]));
  }

  // extent
  static_assert(std::is_same_v<decltype(work_group.extent(0)),
                               typename sycl::khr::work_group<
                                   Dimensions>::extents_type::index_type>);
  for (int i = 0; i < Dimensions; i++)
    passed &= (work_group.extent(i) == work_group.extents().extent(i));

  // rank
  static_assert(
      std::is_same_v<
          decltype(sycl::khr::work_group<Dimensions>::rank()),
          typename sycl::khr::work_group<Dimensions>::extents_type::rank_type>);
  static_assert(decltype(work_group)::rank() ==
                decltype(work_group.extents())::rank());

  // rank_dynamic
  static_assert(
      std::is_same_v<
          decltype(sycl::khr::work_group<Dimensions>::rank_dynamic()),
          typename sycl::khr::work_group<Dimensions>::extents_type::rank_type>);
  static_assert(decltype(work_group)::rank_dynamic() ==
                decltype(work_group.extents())::rank_dynamic());

  // static_extent
  static_assert(std::is_same_v<
                decltype(sycl::khr::work_group<Dimensions>::static_extent(0)),
                std::size_t>);
  static_assert(decltype(work_group)::static_extent(0) ==
                decltype(work_group.extents())::static_extent(0));
  if constexpr (Dimensions >= 2)
    static_assert(decltype(work_group)::static_extent(1) ==
                  decltype(work_group.extents())::static_extent(1));
  if constexpr (Dimensions == 3)
    static_assert(decltype(work_group)::static_extent(2) ==
                  decltype(work_group.extents())::static_extent(2));
#endif

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

#if __cplusplus >= 202302L
  // extents
  static_assert(std::is_same_v<decltype(sub_group.extents()),
                               typename sycl::khr::sub_group::extents_type>);
  passed &= (sub_group.extents() ==
             std::dextents<std::uint32_t, 1>(group.get_local_linear_range()));

  // extent
  static_assert(
      std::is_same_v<decltype(sub_group.extent(0)),
                     typename sycl::khr::sub_group ::extents_type::index_type>);
  passed &= (sub_group.extent(0) == sub_group.extents().extent(0));

  // rank
  static_assert(
      std::is_same_v<decltype(sycl::khr::sub_group::rank()),
                     typename sycl::khr::sub_group::extents_type::rank_type>);
  static_assert(decltype(sub_group)::rank() ==
                decltype(sub_group.extents())::rank());

  // rank_dynamic
  static_assert(
      std::is_same_v<decltype(sycl::khr::sub_group::rank_dynamic()),
                     typename sycl::khr::sub_group::extents_type::rank_type>);
  static_assert(decltype(sub_group)::rank_dynamic() ==
                decltype(sub_group.extents())::rank_dynamic());

  // static_extent
  static_assert(std::is_same_v<decltype(sycl::khr::sub_group::static_extent(0)),
                               std::size_t>);
  static_assert(decltype(sub_group)::static_extent(0) ==
                decltype(sub_group.extents())::static_extent(0));
#endif

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
  sycl::khr::member_item item{sycl::khr::get_member_item(work_group)};

  // id
  static_assert(
      std::is_same_v<decltype(item.id()),
                     typename sycl::khr::member_item<
                         sycl::khr::work_group<Dimensions>>::id_type>);
  passed &= (group.get_local_id() == item.id());

  // linear_id
  static_assert(
      std::is_same_v<decltype(item.linear_id()),
                     typename sycl::khr::member_item<
                         sycl::khr::work_group<Dimensions>>::linear_id_type>);
  passed &= (group.get_local_linear_id() == item.linear_id());

  // range
  static_assert(
      std::is_same_v<decltype(item.range()),
                     typename sycl::khr::member_item<
                         sycl::khr::work_group<Dimensions>>::range_type>);
  passed &= (group.get_local_range() == item.range());

#if __cplusplus >= 202302L
  // extents
  static_assert(
      std::is_same_v<decltype(item.extents()),
                     typename sycl::khr::member_item<
                         sycl::khr::work_group<Dimensions>>::extents_type>);
  if constexpr (Dimensions == 1)
    passed &= (item.extents() == std::extents<std::size_t, 1>());
  else if constexpr (Dimensions == 2)
    passed &= (item.extents() == std::extents<std::size_t, 1, 1>());
  else if constexpr (Dimensions == 3)
    passed &= (item.extents() == std::extents<std::size_t, 1, 1, 1>());

  // extent
  static_assert(
      std::is_same_v<decltype(item.extent(0)),
                     typename sycl::khr::member_item<sycl::khr::work_group<
                         Dimensions>>::extents_type::index_type>);
  for (int i = 0; i < Dimensions; i++)
    passed &= (item.extent(i) == item.extents().extent(i));

  // rank
  static_assert(
      std::is_same_v<decltype(sycl::khr::member_item<
                              sycl::khr::work_group<Dimensions>>::rank()),
                     typename sycl::khr::member_item<sycl::khr::work_group<
                         Dimensions>>::extents_type::rank_type>);
  static_assert(decltype(item)::rank() == decltype(item.extents())::rank());

  // rank_dynamic
  static_assert(
      std::is_same_v<
          decltype(sycl::khr::member_item<
                   sycl::khr::work_group<Dimensions>>::rank_dynamic()),
          typename sycl::khr::member_item<
              sycl::khr::work_group<Dimensions>>::extents_type::rank_type>);
  static_assert(decltype(item)::rank_dynamic() ==
                decltype(item.extents())::rank_dynamic());

  // static_extent
  static_assert(
      std::is_same_v<decltype(sycl::khr::member_item<sycl::khr::work_group<
                                  Dimensions>>::static_extent(0)),
                     std::size_t>);
  static_assert(decltype(item)::static_extent(0) ==
                decltype(item.extents())::static_extent(0));
  if constexpr (Dimensions >= 2)
    static_assert(decltype(item)::static_extent(1) ==
                  decltype(item.extents())::static_extent(1));
  if constexpr (Dimensions == 3)
    static_assert(decltype(item)::static_extent(2) ==
                  decltype(item.extents())::static_extent(2));
#endif

  // size
  static_assert(
      std::is_same_v<decltype(item.size()),
                     typename sycl::khr::member_item<
                         sycl::khr::work_group<Dimensions>>::size_type>);
  passed &= (1 == item.size());

  return passed;
}

template <int Dimensions>
static bool testWorkItemSubgroup(sycl::nd_item<Dimensions> it) {
  bool passed = true;
  sycl::sub_group group{it.get_sub_group()};
  sycl::khr::sub_group sub_group{group};
  sycl::khr::member_item item{sycl::khr::get_member_item(sub_group)};

  // id
  static_assert(
      std::is_same_v<decltype(item.id()), typename sycl::khr::member_item<
                                              sycl::khr::sub_group>::id_type>);
  passed &= (group.get_local_id() == item.id());

  // linear_id
  static_assert(std::is_same_v<decltype(item.linear_id()),
                               typename sycl::khr::member_item<
                                   sycl::khr::sub_group>::linear_id_type>);
  passed &= (group.get_local_linear_id() == item.linear_id());

  // range
  static_assert(
      std::is_same_v<
          decltype(item.range()),
          typename sycl::khr::member_item<sycl::khr::sub_group>::range_type>);
  passed &= (group.get_local_range() == item.range());

#if __cplusplus >= 202302L
  // extents
  static_assert(
      std::is_same_v<
          decltype(item.extents()),
          typename sycl::khr::member_item<sycl::khr::sub_group>::extents_type>);
  passed &= (item.extents() == std::extents<std::uint32_t, 1>());

  // extent
  static_assert(
      std::is_same_v<decltype(item.extent(0)),
                     typename sycl::khr::member_item<
                         sycl::khr::sub_group>::extents_type::index_type>);
  passed &= (item.extent(0) == item.extents().extent(0));

  // rank
  static_assert(std::is_same_v<
                decltype(sycl::khr::member_item<sycl::khr::sub_group>::rank()),
                typename sycl::khr::member_item<
                    sycl::khr::sub_group>::extents_type::rank_type>);
  static_assert(decltype(item)::rank() == decltype(item.extents())::rank());

  // rank_dynamic
  static_assert(
      std::is_same_v<decltype(sycl::khr::member_item<
                              sycl::khr::sub_group>::rank_dynamic()),
                     typename sycl::khr::member_item<
                         sycl::khr::sub_group>::extents_type::rank_type>);
  static_assert(decltype(item)::rank_dynamic() ==
                decltype(item.extents())::rank_dynamic());

  // static_extent
  static_assert(
      std::is_same_v<decltype(sycl::khr::member_item<sycl::khr::work_group<
                                  Dimensions>>::static_extent(0)),
                     std::size_t>);
  static_assert(decltype(item)::static_extent(0) ==
                decltype(item.extents())::static_extent(0));
#endif

  // size
  static_assert(
      std::is_same_v<
          decltype(item.size()),
          typename sycl::khr::member_item<sycl::khr::sub_group>::size_type>);
  passed &= (1 == item.size());

  return passed;
}

template <int Dimensions>
static void testGroupInterface(sycl::nd_range<Dimensions> nd_range) {
  constexpr std::size_t test_count = 4;
  sycl::queue q{sycl_cts::util::get_cts_object::queue()};
  sycl::buffer<bool> results{nd_range.get_global_range().size() * test_count};
  results.get_range();
  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      acc[(it.get_global_linear_id() * test_count) + 0] = testWorkGroup(it);
      acc[(it.get_global_linear_id() * test_count) + 1] = testSubGroup(it);
      acc[(it.get_global_linear_id() * test_count) + 2] = testWorkItemGroup(it);
      acc[(it.get_global_linear_id() * test_count) + 3] =
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
