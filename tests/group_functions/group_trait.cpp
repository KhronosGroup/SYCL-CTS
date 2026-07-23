/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"

#include "group_functions_common.h"

TEST_CASE("Group type trait", "[group_func]") {
  // positive testing
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::group<1>>>);
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::group<2>>>);
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::group<3>>>);
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::sub_group>>);

  // negative testing
  CHECK(std::is_base_of_v<std::false_type, sycl::is_group<sycl::nd_range<3>>>);

  // positive testing
  CHECK(sycl::is_group_v<sycl::group<1>>);
  CHECK(sycl::is_group_v<sycl::group<2>>);
  CHECK(sycl::is_group_v<sycl::group<3>>);
  CHECK(sycl::is_group_v<sycl::sub_group>);

  // negative testing
  CHECK_FALSE(sycl::is_group_v<sycl::nd_range<2>>);
}
