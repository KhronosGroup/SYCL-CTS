/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include <catch2/catch_template_test_macros.hpp>

#include "group_barrier.h"

namespace non_uniform_groups::tests {

template <int D>
class test_fence;

TEMPLATE_LIST_TEST_CASE("Non-uniform-group barriers",
                        "[oneapi_non_uniform_groups][group_func]",
                        GroupPackTypes) {
  for_all_combinations<non_uniform_group_barrier_test>(TestType{});
}

}  // namespace non_uniform_groups::tests
