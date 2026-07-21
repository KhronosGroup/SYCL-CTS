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

TEST_CASE("Test for is_fixed_topology_group trait with existing groups.",
          "[oneapi_non_uniform_groups]") {
#ifndef SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS
  SKIP("SYCL_EXT_ONEAPI_NON_UNIFORM_GROUPS is not defined");
#else
  namespace oneapi_ext = sycl::ext::oneapi::experimental;

  STATIC_CHECK(oneapi_ext::is_fixed_topology_group<sycl::group<1>>::value);
  STATIC_CHECK(oneapi_ext::is_fixed_topology_group_v<sycl::group<1>>);
  STATIC_CHECK(oneapi_ext::is_fixed_topology_group<sycl::group<2>>::value);
  STATIC_CHECK(oneapi_ext::is_fixed_topology_group_v<sycl::group<2>>);
  STATIC_CHECK(oneapi_ext::is_fixed_topology_group<sycl::group<3>>::value);
  STATIC_CHECK(oneapi_ext::is_fixed_topology_group_v<sycl::group<3>>);

  STATIC_CHECK(oneapi_ext::is_fixed_topology_group<sycl::sub_group>::value);
  STATIC_CHECK(oneapi_ext::is_fixed_topology_group_v<sycl::sub_group>);

#ifdef SYCL_EXT_ONEAPI_ROOT_GROUP
  STATIC_CHECK(
      oneapi_ext::is_fixed_topology_group<oneapi_ext::root_group<1>>::value);
  STATIC_CHECK(
      oneapi_ext::is_fixed_topology_group_v<oneapi_ext::root_group<1>>);
  STATIC_CHECK(
      oneapi_ext::is_fixed_topology_group<oneapi_ext::root_group<2>>::value);
  STATIC_CHECK(
      oneapi_ext::is_fixed_topology_group_v<oneapi_ext::root_group<2>>);
  STATIC_CHECK(
      oneapi_ext::is_fixed_topology_group<oneapi_ext::root_group<3>>::value);
  STATIC_CHECK(
      oneapi_ext::is_fixed_topology_group_v<oneapi_ext::root_group<3>>);
#endif
#endif
}

}  // namespace non_uniform_groups::tests
