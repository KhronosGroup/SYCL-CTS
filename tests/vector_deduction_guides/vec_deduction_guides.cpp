/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides vec deduction guides tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/type_list.h"

namespace vec_deduction_guides {
using namespace sycl;

template <typename T>
class check_vec_deduction {
 public:
  void operator()(const std::string& type) {
    type_name = type;

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
  std::string type_name;

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
           type_name + ", " + std::to_string(vector.size()) + "> ");
      CHECK(vector[i] == static_cast<T>(i));
    }
  }
};

// FIXME: re-enable when vec deduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("vec deduction guides", "[vec_deduction]")({
  for_all_types<check_vec_deduction>(deduction::vector_types);
});
}  // namespace vec_deduction_guides
