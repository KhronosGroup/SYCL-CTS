/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/
#include "catch2/catch_test_macros.hpp"

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../common/range_index_space_id.h"

namespace item_members {

template <int Dim>
void run_test() {
  std::string type_name =
      std::string("sycl::item<") + std::to_string(Dim) + ">";
  range_index_space_id::check_members_test<sycl::item<Dim>, Dim>(type_name);
}

TEST_CASE("sycl::item members", "[item]") {
  run_test<1>();
  run_test<2>();
  run_test<3>();
}

}  // namespace item_members
