/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../common/range_index_space_id.h"

namespace sub_group_members {

void run_test() {
  std::string type_name = std::string("sycl::sub_group");
  range_index_space_id::check_members_test<sycl::sub_group, 1>(type_name);
}

TEST_CASE("sycl::sub_group members", "[sub_group]") { run_test(); }

}  // namespace sub_group_members
