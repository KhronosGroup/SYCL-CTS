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
//  Provides test for the sycl::bit_cast.
//
*******************************************************************************/
#ifndef SYCL_CTS_BIT_CAST_TEST_H
#define SYCL_CTS_BIT_CAST_TEST_H

#include "../common/once_per_unit.h"
#include "bit_cast_helper_functions.h"
#include <cstring>

namespace bit_cast::tests {
using namespace bit_cast::tests::helper_functions;

constexpr int expected_val = 42;

template <typename ToType, typename FromType>
class kernel_name;

template <typename ToType, typename FromType>
class bit_cast_test {
  static constexpr bool is_invalid_test_case =
      std::is_array_v<ToType> || sizeof(ToType) != sizeof(FromType) ||
      std::is_same_v<ToType, bool>;

 public:
  void operator()(const std::string& to_type_name,
                  const std::string& from_type_name) {
    auto queue = once_per_unit::get_queue();
    if constexpr (is_invalid_test_case)
      return;
    else {
      std::array test_result{false, false};
      {
        sycl::buffer<bool, 1> buf_result(test_result.data(),
                                         sycl::range<1>(test_result.size()));
        queue
            .submit([&](sycl::handler& cgh) {
              auto result_acc =
                  buf_result.template get_access<sycl::access_mode::write>(cgh);
              cgh.single_task<kernel_name<ToType, FromType>>([=] {
                FromType expected;
                value_operations::assign(expected, expected_val);
                FromType from;
                value_operations::assign(from, expected_val);
                auto to = sycl::bit_cast<ToType>(from);
                result_acc[0] =
                    memcmp_no_ext_lib(&to, &from, sizeof(from)) == 0;
                if constexpr (!std::is_array_v<FromType>) {
                  from = sycl::bit_cast<FromType>(to);
                  result_acc[1] = value_operations::are_equal(from, expected);
                }
              });
            })
            .wait_and_throw();
      }
      {
        INFO("Memory contents are not equal. "
             << "ToType : " << to_type_name << " FromType: " << from_type_name);
        CHECK(test_result[0]);
      }
      if constexpr (!std::is_array_v<FromType>) {
        INFO("Round trip conversion failed. "
             << "ToType : " << to_type_name << " FromType: " << from_type_name);
        CHECK(test_result[1]);
      }
    }
  }
};

template <typename PrimaryTypeTo, typename PrimaryTypeFrom>
class run_bit_cast_test {
 public:
  void operator()(const std::string& primary_to_type_name,
                  const std::string& primary_from_type_name) {
    const auto to_types =
        get_derived_type_pack<PrimaryTypeTo>(primary_to_type_name);
    const auto from_types =
        get_derived_type_pack<PrimaryTypeFrom>(primary_from_type_name);
    const auto to_types_ptrs =
        get_derived_type_pack<PrimaryTypeTo*>(primary_to_type_name + "*");
    const auto from_types_ptrs =
        get_derived_type_pack<PrimaryTypeFrom*>(primary_from_type_name + "*");

    for_all_combinations<bit_cast_test>(to_types, from_types);
    for_all_combinations<bit_cast_test>(to_types_ptrs, from_types);
    for_all_combinations<bit_cast_test>(to_types, from_types_ptrs);
    for_all_combinations<bit_cast_test>(to_types_ptrs, from_types_ptrs);
  }
};

}  // namespace bit_cast::tests

#endif  // SYCL_CTS_BIT_CAST_TEST_H
