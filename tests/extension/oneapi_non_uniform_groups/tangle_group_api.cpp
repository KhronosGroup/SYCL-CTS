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

TEST_CASE("Test for tangle_group apis.", "[oneapi_non_uniform_groups]") {
#ifndef SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS
  SKIP("SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS is not defined");
#else
  namespace oneapi_ext = sycl::ext::oneapi::experimental;
  using tangle_group_t = oneapi_ext::tangle_group<sycl::sub_group>;
  using CheckResults = bool[checks::COUNT];

  constexpr size_t num_items = 64;

  sycl::buffer<CheckResults, 1> results_buffer{num_items};

  auto q = sycl_cts::util::get_cts_object::queue();

  if (!q.get_device().has(sycl::aspect::ext_oneapi_tangle_group)) {
    SKIP("Device does not support tangle_group.");
  }

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results_buffer, cgh, sycl::write_only};

    sycl::nd_range<1> nd_range{sycl::range<1>{num_items},
                               sycl::range<1>{num_items}};

    cgh.parallel_for<struct TangleGroupAPI>(nd_range, [=](sycl::nd_item<1> it) {
      auto& results = acc[it.get_global_id()];

      sycl::sub_group sg = it.get_sub_group();
      size_t sub_group_size = sg.get_local_range().size();
      size_t split = sub_group_size / 3;

      auto run_checks = [&](tangle_group_t tangle, size_t expected_size) {
        results[checks::get_group_id] = tangle.get_group_id() == 0;
        results[checks::get_local_id] = tangle.get_local_id() < split;
        results[checks::get_group_range] = tangle.get_group_range().size() == 1;
        results[checks::get_local_range] =
            tangle.get_local_range().size() == expected_size;
        results[checks::get_group_linear_id] =
            tangle.get_group_linear_id() == 0;
        results[checks::get_local_linear_id] =
            tangle.get_local_linear_id() == tangle.get_local_id();
        results[checks::get_group_linear_range] =
            tangle.get_group_linear_range() == tangle.get_group_range().size();
        results[checks::get_local_linear_range] =
            tangle.get_local_linear_range() == tangle.get_local_range().size();
        results[checks::leader] =
            tangle.leader() == (tangle.get_local_id() == 0);
      };

      if (sg.get_local_linear_id() < split) {
        auto tangle = oneapi_ext::get_tangle_group(sg);
        static_assert(std::is_same_v<decltype(tangle), tangle_group_t>);
        run_checks(tangle, split);
      } else {
        auto tangle = oneapi_ext::get_tangle_group(sg);
        static_assert(std::is_same_v<decltype(tangle), tangle_group_t>);
        run_checks(tangle, sub_group_size - split);
      }
    });
  });

  CheckResults results = {};
  sycl::accessor acc = results_buffer.get_host_access();
  for (size_t check = 0; check < checks::COUNT; check++)
    results[check] = std::all_of(acc.cbegin(), acc.cend(),
                                 [=](const auto& it) { return it[check]; });

  // Group-category traits.
  STATIC_CHECK(sycl::is_group<tangle_group_t>::value);
  STATIC_CHECK(sycl::is_group_v<tangle_group_t>);
  STATIC_CHECK(oneapi_ext::is_user_constructed_group<tangle_group_t>::value);
  STATIC_CHECK(oneapi_ext::is_user_constructed_group_v<tangle_group_t>);
  STATIC_CHECK(!oneapi_ext::is_fixed_topology_group<tangle_group_t>::value);
  STATIC_CHECK(!oneapi_ext::is_fixed_topology_group_v<tangle_group_t>);

  // Aliases.
  STATIC_CHECK(std::is_same_v<tangle_group_t::id_type, sycl::id<1>>);
  STATIC_CHECK(std::is_same_v<tangle_group_t::range_type, sycl::range<1>>);
  STATIC_CHECK(std::is_same_v<tangle_group_t::linear_id_type, uint32_t>);

  // Static constexpr members.
  STATIC_CHECK(tangle_group_t::dimensions == 1);
  STATIC_CHECK(tangle_group_t::fence_scope == sycl::sub_group::fence_scope);

  // get_group_id
  CHECK(std::is_same_v<decltype(std::declval<tangle_group_t>().get_group_id()),
                       tangle_group_t::id_type>);
  CHECK(results[checks::get_group_id]);

  // get_local_id
  CHECK(std::is_same_v<decltype(std::declval<tangle_group_t>().get_local_id()),
                       tangle_group_t::id_type>);
  CHECK(results[checks::get_local_id]);

  // get_group_range
  CHECK(
      std::is_same_v<decltype(std::declval<tangle_group_t>().get_group_range()),
                     tangle_group_t::range_type>);
  CHECK(results[checks::get_group_range]);

  // get_local_range
  CHECK(
      std::is_same_v<decltype(std::declval<tangle_group_t>().get_local_range()),
                     tangle_group_t::range_type>);
  CHECK(results[checks::get_local_range]);

  // get_group_linear_id
  CHECK(std::is_same_v<
        decltype(std::declval<tangle_group_t>().get_group_linear_id()),
        tangle_group_t::linear_id_type>);
  CHECK(results[checks::get_group_linear_id]);

  // get_local_linear_id
  CHECK(std::is_same_v<
        decltype(std::declval<tangle_group_t>().get_local_linear_id()),
        tangle_group_t::linear_id_type>);
  CHECK(results[checks::get_local_linear_id]);

  // get_group_linear_range
  CHECK(std::is_same_v<
        decltype(std::declval<tangle_group_t>().get_group_linear_range()),
        tangle_group_t::linear_id_type>);
  CHECK(results[checks::get_group_linear_range]);

  // get_local_linear_range
  CHECK(std::is_same_v<
        decltype(std::declval<tangle_group_t>().get_local_linear_range()),
        tangle_group_t::linear_id_type>);
  CHECK(results[checks::get_local_linear_range]);

  // leader
  CHECK(
      std::is_same_v<decltype(std::declval<tangle_group_t>().leader()), bool>);
  CHECK(results[checks::leader]);
#endif
}

}  // namespace non_uniform_groups::tests
