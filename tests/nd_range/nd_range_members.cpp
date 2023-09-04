/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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
#include "catch2/catch_test_macros.hpp"

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../common/range_index_space_id.h"

namespace nd_range_members {

template <int Dim>
void run_test() {
  std::string type_name =
      std::string("sycl::nd_range<") + std::to_string(Dim) + ">";
  range_index_space_id::check_members_test<sycl::nd_range<Dim>, Dim>(type_name);
}

TEST_CASE("sycl::nd_range members", "[nd_range]") {
  run_test<1>();
  run_test<2>();
  run_test<3>();
}

}  // namespace nd_range_members
