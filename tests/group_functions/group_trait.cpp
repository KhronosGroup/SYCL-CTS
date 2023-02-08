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

#include "../common/common.h"

#include "group_functions_common.h"

TEST_CASE("Group type trait", "[group_func]") {
#ifdef SYCL_CTS_COMPILING_WITH_DPCPP
  WARN("DPCPP does not implement sycl::is_group. Skipping the test case.");
#else
  // positive testing
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::group<1>>>);
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::group<2>>>);
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::group<3>>>);
  CHECK(std::is_base_of_v<std::true_type, sycl::is_group<sycl::sub_group>>);

  // negative testing
  CHECK(std::is_base_of_v<std::false_type, sycl::is_group<sycl::nd_range<3>>>);
#endif

  // positive testing
  CHECK(sycl::is_group_v<sycl::group<1>>);
  CHECK(sycl::is_group_v<sycl::group<2>>);
  CHECK(sycl::is_group_v<sycl::group<3>>);
  CHECK(sycl::is_group_v<sycl::sub_group>);

  // negative testing
  CHECK_FALSE(sycl::is_group_v<sycl::nd_range<2>>);
}
