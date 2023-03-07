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
//  Provides id deduction guides tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_list.h"

namespace id_deduction_guides {
using namespace sycl;

// array with sizes
constexpr std::size_t N[3] = {4, 8, 10};

template <int dims, class idT>
void check_id_operator(idT _id) {
  for (int i = 0; i < dims; ++i) {
    INFO("operator[] returns wrong value with id<" + std::to_string(dims) +
         ">");
    CHECK(_id[i] == N[i]);
  }
}

template <int dims, class idT>
void check_id_type(idT _id) {
  INFO("Wrong id type, expected id<" + std::to_string(dims) + ">");
  CHECK(std::is_same_v<idT, id<dims>>);
}

TEST_CASE("id deduction guides", "[id]") {
  id id_1d(N[0]);
  id id_2d(N[0], N[1]);
  id id_3d(N[0], N[1], N[2]);

  check_id_operator<1>(id_1d);
  check_id_operator<2>(id_2d);
  check_id_operator<3>(id_3d);

  check_id_type<1>(id_1d);
  check_id_type<2>(id_2d);
  check_id_type<3>(id_3d);
}
}  // namespace id_deduction_guides
