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
//  Provides vec deduction guides tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_list.h"
#include "../common/disabled_for_test_case.h"

namespace vec_deduction_guides {
using namespace sycl;

template <typename T>
class check_vec_deduction {
 public:
  void operator()(const std::string& type) {
    typeName = type;

    T data[max_size];
    for (int i = 0; i < max_size; ++i) {
      data[i] = static_cast<T>(i);
    }

    vec vec_1n(data[0]);
    vec vec_2n(data[0], data[1]);
    vec vec_3n(data[0], data[1], data[2]);
    vec vec_4n(data[0], data[1], data[2], data[3]);
    vec vec_8n(data[0], data[1], data[2], data[3], data[4], data[5], data[6],
               data[7]);
    vec vec_16n(data[0], data[1], data[2], data[3], data[4], data[5], data[6],
                data[7], data[8], data[9], data[10], data[11], data[12],
                data[13], data[14], data[15]);

    check_vector_value(vec_1n);
    check_vector_value(vec_2n);
    check_vector_value(vec_3n);
    check_vector_value(vec_4n);
    check_vector_value(vec_8n);
    check_vector_value(vec_16n);

    check_vector_type<1>(vec_1n);
    check_vector_type<2>(vec_2n);
    check_vector_type<3>(vec_3n);
    check_vector_type<4>(vec_4n);
    check_vector_type<8>(vec_8n);
    check_vector_type<16>(vec_16n);
  }

 private:
  const int max_size = 16;
  std::string typeName;

  template <int expected_size, class vecT>
  void check_vector_type(vecT vector) {
    INFO("Wrong vec type");
    CHECK(std::is_same_v<vecT, vec<T, expected_size>>);
  }

  template <class vecT>
  void check_vector_value(vecT vector) {
    for (int i = 0; i < vector.size(); ++i) {
      INFO("Wrong vec value on index " + std::to_string(i) +
           " with value: " + std::to_string(vector[i]) + " vec type: vec<" +
           typeName + ", " + std::to_string(vector.size()) + "> ");
      CHECK(vector[i] == static_cast<T>(i));
    }
  }
};

// FIXME: re-enable when vec deduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
("vec deduction guides", "[vec_deduction]")({
  for_all_types<check_vec_deduction>(deduction::vector_types);
});
}  // namespace vec_deduction_guides
