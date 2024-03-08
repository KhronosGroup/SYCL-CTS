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

#include <catch2/catch_template_test_macros.hpp>

#include "group_barrier.h"

template <int D>
class test_fence;

TEST_CASE("Non-uniform-group barriers",
          "[oneapi_non_uniform_groups][group_func]") {
  auto queue = once_per_unit::get_queue();

  non_uniform_group_barrier<oneapi_ext::ballot_group<sycl::sub_group>>(queue);
  non_uniform_group_barrier<oneapi_ext::fixed_size_group<1, sycl::sub_group>>(
      queue);
  non_uniform_group_barrier<oneapi_ext::fixed_size_group<2, sycl::sub_group>>(
      queue);
  non_uniform_group_barrier<oneapi_ext::fixed_size_group<4, sycl::sub_group>>(
      queue);
  non_uniform_group_barrier<oneapi_ext::fixed_size_group<8, sycl::sub_group>>(
      queue);
  non_uniform_group_barrier<oneapi_ext::tangle_group<sycl::sub_group>>(queue);
  non_uniform_group_barrier<oneapi_ext::opportunistic_group>(queue);
}
