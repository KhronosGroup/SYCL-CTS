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
