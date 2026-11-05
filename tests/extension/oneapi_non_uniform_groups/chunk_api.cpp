/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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

// TODO: Parameterize the tests on this and test edge-cases, like 1 and
//       the sub-group size.
constexpr size_t chunk_size = 4;

TEST_CASE("Test for chunk apis.", "[oneapi_non_uniform_groups]") {
#ifndef SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS
  SKIP("SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS is not defined");
#else
  namespace oneapi_ext = sycl::ext::oneapi::experimental;
  using chunk_t = oneapi_ext::chunk<chunk_size, sycl::sub_group>;
  using CheckResults = bool[checks::COUNT];

  constexpr size_t num_items = 64;

  sycl::buffer<CheckResults, 1> results_buffer{num_items};

  auto q = sycl_cts::util::get_cts_object::queue();

  if (!q.get_device().has(sycl::aspect::ext_oneapi_chunk)) {
    SKIP("Device does not support chunk.");
  }

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{results_buffer, cgh, sycl::write_only};

    sycl::nd_range<1> nd_range{sycl::range<1>{num_items},
                               sycl::range<1>{num_items}};

    cgh.parallel_for<struct ChunkAPI>(nd_range, [=](sycl::nd_item<1> it) {
      auto& results = acc[it.get_global_id()];

      sycl::sub_group sg = it.get_sub_group();
      size_t sub_group_size = sg.get_local_range().size();

      auto chunk_group = oneapi_ext::chunked_partition<chunk_size>(sg);
      static_assert(std::is_same_v<decltype(chunk_group), chunk_t>);

      results[checks::get_group_id] =
          chunk_group.get_group_id() < sub_group_size / chunk_size;
      results[checks::get_local_id] = chunk_group.get_local_id() < chunk_size;
      results[checks::get_group_range] =
          chunk_group.get_group_range().size() == sub_group_size / chunk_size;
      results[checks::get_local_range] =
          chunk_group.get_local_range().size() == chunk_size;
      results[checks::get_group_linear_id] =
          chunk_group.get_group_linear_id() == chunk_group.get_group_id();
      results[checks::get_local_linear_id] =
          chunk_group.get_local_linear_id() == chunk_group.get_local_id();
      results[checks::get_group_linear_range] =
          chunk_group.get_group_linear_range() ==
          chunk_group.get_group_range().size();
      results[checks::get_local_linear_range] =
          chunk_group.get_local_linear_range() ==
          chunk_group.get_local_range().size();
      results[checks::leader] =
          chunk_group.leader() == (chunk_group.get_local_id() == 0);
    });
  });

  CheckResults results = {};
  sycl::accessor acc = results_buffer.get_host_access();
  for (size_t check = 0; check < checks::COUNT; check++)
    results[check] = std::all_of(acc.cbegin(), acc.cend(),
                                 [=](const auto& it) { return it[check]; });

  // Group-category traits.
  STATIC_CHECK(sycl::is_group<chunk_t>::value);
  STATIC_CHECK(sycl::is_group_v<chunk_t>);
  STATIC_CHECK(oneapi_ext::is_user_constructed_group<chunk_t>::value);
  STATIC_CHECK(oneapi_ext::is_user_constructed_group_v<chunk_t>);
  STATIC_CHECK(!oneapi_ext::is_fixed_topology_group<chunk_t>::value);
  STATIC_CHECK(!oneapi_ext::is_fixed_topology_group_v<chunk_t>);

  // Aliases.
  STATIC_CHECK(std::is_same_v<chunk_t::id_type, sycl::id<1>>);
  STATIC_CHECK(std::is_same_v<chunk_t::range_type, sycl::range<1>>);
  STATIC_CHECK(std::is_same_v<chunk_t::linear_id_type, uint32_t>);

  // Static constexpr members.
  STATIC_CHECK(chunk_t::dimensions == 1);
  STATIC_CHECK(chunk_t::fence_scope == sycl::sub_group::fence_scope);

  // get_group_id
  CHECK(std::is_same_v<decltype(std::declval<chunk_t>().get_group_id()),
                       chunk_t::id_type>);
  CHECK(results[checks::get_group_id]);

  // get_local_id
  CHECK(std::is_same_v<decltype(std::declval<chunk_t>().get_local_id()),
                       chunk_t::id_type>);
  CHECK(results[checks::get_local_id]);

  // get_group_range
  CHECK(std::is_same_v<decltype(std::declval<chunk_t>().get_group_range()),
                       chunk_t::range_type>);
  CHECK(results[checks::get_group_range]);

  // get_local_range
  CHECK(std::is_same_v<decltype(std::declval<chunk_t>().get_local_range()),
                       chunk_t::range_type>);
  CHECK(results[checks::get_local_range]);

  // get_group_linear_id
  CHECK(std::is_same_v<decltype(std::declval<chunk_t>().get_group_linear_id()),
                       chunk_t::linear_id_type>);
  CHECK(results[checks::get_group_linear_id]);

  // get_local_linear_id
  CHECK(std::is_same_v<decltype(std::declval<chunk_t>().get_local_linear_id()),
                       chunk_t::linear_id_type>);
  CHECK(results[checks::get_local_linear_id]);

  // get_group_linear_range
  CHECK(
      std::is_same_v<decltype(std::declval<chunk_t>().get_group_linear_range()),
                     chunk_t::linear_id_type>);
  CHECK(results[checks::get_group_linear_range]);

  // get_local_linear_range
  CHECK(
      std::is_same_v<decltype(std::declval<chunk_t>().get_local_linear_range()),
                     chunk_t::linear_id_type>);
  CHECK(results[checks::get_local_linear_range]);

  // leader
  CHECK(std::is_same_v<decltype(std::declval<chunk_t>().leader()), bool>);
  CHECK(results[checks::leader]);
#endif
}

}  // namespace non_uniform_groups::tests
