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
//   Provides range deduction guides tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_list.h"

namespace range_deduction_guides {
using namespace sycl;

// array with sizes
constexpr std::size_t N[3] = {4, 8, 10};

template <int dims, class rangeT>
void check_range_size(rangeT r) {
  std::size_t expected_size = 1;
  for (int i = 0; i < dims; ++i) {
    expected_size *= N[i];
  }

  INFO("Wrong range size, expected " + std::to_string(expected_size));
  CHECK(expected_size == r.size());
}

template <int dims, class rangeT>
void check_range_operator(rangeT r) {
  for (int i = 0; i < dims; ++i) {
    INFO("operator[] returns wrong value with range<" + std::to_string(dims) +
         ">");
    CHECK(r[i] == N[i]);
  }
}

template <int dims, class rangeT>
void check_range_type(rangeT r) {
  INFO("Wrong range type, expected range<" + std::to_string(dims) + ">");
  CHECK(std::is_same_v<rangeT, range<dims>>);
}

TEST_CASE("range deduction guides", "[range]") {
  range range_1d(N[0]);
  range range_2d(N[0], N[1]);
  range range_3d(N[0], N[1], N[2]);

  check_range_size<1>(range_1d);
  check_range_size<2>(range_2d);
  check_range_size<3>(range_3d);

  check_range_operator<1>(range_1d);
  check_range_operator<2>(range_2d);
  check_range_operator<3>(range_3d);

  check_range_type<1>(range_1d);
  check_range_type<2>(range_2d);
  check_range_type<3>(range_3d);
}
}  // namespace range_deduction_guides
