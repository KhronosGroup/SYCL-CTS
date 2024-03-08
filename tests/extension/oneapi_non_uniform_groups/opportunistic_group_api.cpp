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

#include "../../common/common.h"

namespace non_uniform_groups::tests {

struct checks {
  enum {
    get_group_id,
    get_local_id,
    get_group_range,
    get_local_range,
    get_group_linear_id,
    get_local_linear_id,
    get_group_linear_range,
    get_local_linear_range,
    leader,
    COUNT,
  };
};

TEST_CASE("Test for opportunistic_group apis.", "[oneapi_non_uniform_groups]") {
#ifndef SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS
  SKIP("SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS is not defined");
#else
  namespace oneapi_ext = sycl::ext::oneapi::experimental;
  using opportunistic_group_t = oneapi_ext::opportunistic_group;
  using CheckResults = bool[checks::COUNT];

  constexpr size_t num_items = 64;

  sycl::buffer<CheckResults, 1> results_buffer{num_items};

  auto q = sycl_cts::util::get_cts_object::queue();

  if (!q.get_device().has(sycl::aspect::ext_oneapi_opportunistic_group)) {
    SKIP("Device does not support opportunistic_group.");
  }

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results_buffer, cgh, sycl::write_only};

    sycl::nd_range<1> nd_range{sycl::range<1>{num_items},
                               sycl::range<1>{num_items}};

    cgh.parallel_for<struct OpportunisticGroupAPI>(
        nd_range, [=](sycl::nd_item<1> it) {
          auto& results = acc[it.get_global_id()];

          size_t sub_group_size = it.get_sub_group().get_local_range().size();

          auto opportunistic =
              oneapi_ext::this_kernel::get_opportunistic_group();
          static_assert(
              std::is_same_v<decltype(opportunistic), opportunistic_group_t>);

          results[checks::get_group_id] = opportunistic.get_group_id() == 0;
          results[checks::get_local_id] =
              opportunistic.get_local_id() <
              opportunistic.get_local_range().size();
          results[checks::get_group_range] =
              opportunistic.get_group_range().size() == 1;
          results[checks::get_local_range] =
              opportunistic.get_local_range().size() <= sub_group_size;
          results[checks::get_group_linear_id] =
              opportunistic.get_group_linear_id() == 0;
          results[checks::get_local_linear_id] =
              opportunistic.get_local_linear_id() ==
              opportunistic.get_local_id();
          results[checks::get_group_linear_range] =
              opportunistic.get_group_linear_range() ==
              opportunistic.get_group_range().size();
          results[checks::get_local_linear_range] =
              opportunistic.get_local_linear_range() ==
              opportunistic.get_local_range().size();
          results[checks::leader] =
              opportunistic.leader() == (opportunistic.get_local_id() == 0);
        });
  });
  q.wait();

  CheckResults results = {};
  sycl::accessor acc = results_buffer.get_host_access();
  for (size_t check = 0; check < checks::COUNT; check++)
    results[check] = std::all_of(acc.cbegin(), acc.cend(),
                                 [=](const auto& it) { return it[check]; });

  // Group-category traits.
  STATIC_CHECK(sycl::is_group<opportunistic_group_t>::value);
  STATIC_CHECK(sycl::is_group_v<opportunistic_group_t>);
  STATIC_CHECK(
      oneapi_ext::is_user_constructed_group<opportunistic_group_t>::value);
  STATIC_CHECK(oneapi_ext::is_user_constructed_group_v<opportunistic_group_t>);
  STATIC_CHECK(
      !oneapi_ext::is_fixed_topology_group<opportunistic_group_t>::value);
  STATIC_CHECK(!oneapi_ext::is_fixed_topology_group_v<opportunistic_group_t>);

  // Aliases.
  STATIC_CHECK(std::is_same_v<opportunistic_group_t::id_type, sycl::id<1>>);
  STATIC_CHECK(
      std::is_same_v<opportunistic_group_t::range_type, sycl::range<1>>);
  STATIC_CHECK(std::is_same_v<opportunistic_group_t::linear_id_type, uint32_t>);

  // Static constexpr members.
  STATIC_CHECK(opportunistic_group_t::dimensions == 1);
  STATIC_CHECK(opportunistic_group_t::fence_scope ==
               sycl::sub_group::fence_scope);

  // get_group_id
  CHECK(std::is_same_v<
        decltype(std::declval<opportunistic_group_t>().get_group_id()),
        opportunistic_group_t::id_type>);
  CHECK(results[checks::get_group_id]);

  // get_local_id
  CHECK(std::is_same_v<
        decltype(std::declval<opportunistic_group_t>().get_local_id()),
        opportunistic_group_t::id_type>);
  CHECK(results[checks::get_local_id]);

  // get_group_range
  CHECK(std::is_same_v<
        decltype(std::declval<opportunistic_group_t>().get_group_range()),
        opportunistic_group_t::range_type>);
  CHECK(results[checks::get_group_range]);

  // get_local_range
  CHECK(std::is_same_v<
        decltype(std::declval<opportunistic_group_t>().get_local_range()),
        opportunistic_group_t::range_type>);
  CHECK(results[checks::get_local_range]);

  // get_group_linear_id
  CHECK(std::is_same_v<
        decltype(std::declval<opportunistic_group_t>().get_group_linear_id()),
        opportunistic_group_t::linear_id_type>);
  CHECK(results[checks::get_group_linear_id]);

  // get_local_linear_id
  CHECK(std::is_same_v<
        decltype(std::declval<opportunistic_group_t>().get_local_linear_id()),
        opportunistic_group_t::linear_id_type>);
  CHECK(results[checks::get_local_linear_id]);

  // get_group_linear_range
  CHECK(std::is_same_v<decltype(std::declval<opportunistic_group_t>()
                                    .get_group_linear_range()),
                       opportunistic_group_t::linear_id_type>);
  CHECK(results[checks::get_group_linear_range]);

  // get_local_linear_range
  CHECK(std::is_same_v<decltype(std::declval<opportunistic_group_t>()
                                    .get_local_linear_range()),
                       opportunistic_group_t::linear_id_type>);
  CHECK(results[checks::get_local_linear_range]);

  // leader
  CHECK(std::is_same_v<decltype(std::declval<opportunistic_group_t>().leader()),
                       bool>);
  CHECK(results[checks::leader]);
#endif
}

}  // namespace non_uniform_groups::tests
