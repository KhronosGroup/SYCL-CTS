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

#ifndef __SYCLCTS_TESTS_COMMON_RANGE_INDEX_SPACE_ID_H
#define __SYCLCTS_TESTS_COMMON_RANGE_INDEX_SPACE_ID_H

#ifndef SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "../common/type_coverage.h"
#endif

#include "once_per_unit.h"

namespace range_index_space_id {

enum class check_codes : size_t {
  dimensions_type = 0,
  dimensions_value = 1,
  code_count = 2,
};

constexpr size_t check_count = to_integral(check_codes::code_count);

static const std::array<std::string, check_count> error_strings{
    "Wrong type of member T<Dim>::dimensions",
    "Wrong value of member T<Dim>::dimensions",
};

template <check_codes Code, typename ResultArray>
void set_success_operation(ResultArray& result) {
  int index = to_integral(Code);
  result[index] = true;
}

inline std::string get_error_string(int code, const std::string& type_name) {
  return error_strings[code] + " for type T '" + type_name + "'";
}

template <typename T, int Dim, typename ResultArray>
void check_members(ResultArray& result) {
  if (std::is_same_v<const int, decltype(T::dimensions)>) {
    set_success_operation<check_codes::dimensions_type>(result);
  }
  if (Dim == T::dimensions) {
    set_success_operation<check_codes::dimensions_value>(result);
  }
}

template <typename T, int Dim>
class kernel_range_id;

template <typename T, int Dim>
void check_members_test(const std::string& type_name) {
  auto queue = once_per_unit::get_queue();
  std::string section_name = std::string("Checking for type '") + type_name +
                             "' and dim = " + std::to_string(Dim) +
                             " in kernel function";
  bool result[check_count];
  std::fill(result, result + check_count, false);
  SECTION(section_name) {
    {
      sycl::buffer<bool, 1> res_buf(result, sycl::range(check_count));
      queue.submit([&](sycl::handler& cgh) {
        sycl::accessor res_acc(res_buf, cgh);
        cgh.single_task<kernel_range_id<T, Dim>>(
            [=] { check_members<T, Dim>(res_acc); });
      });
    }
    for (size_t i = 0; i < check_count; ++i) {
      INFO(get_error_string(i, type_name));
      CHECK(result[i]);
    }
  }
  section_name = std::string("Checking for type '") + type_name +
                 "' and dim = " + std::to_string(Dim) + " on host";
  SECTION(section_name) {
    check_members<T, Dim>(result);
    for (size_t i = 0; i < check_count; ++i) {
      INFO(get_error_string(i, type_name));
      CHECK(result[i]);
    }
  }
}

}  // namespace range_index_space_id

/// Linearizes a multi-dimensional index according to the specification.
template <unsigned int dimension>
size_t linearize(sycl::range<dimension> range, sycl::id<dimension> id);

inline size_t linearize(sycl::range<1> range, sycl::id<1> id) {
  static_cast<void>(range);
  return id[0];
}

inline size_t linearize(sycl::range<2> range, sycl::id<2> id) {
  return id[1] + id[0] * range[1];
}

inline size_t linearize(sycl::range<3> range, sycl::id<3> id) {
  return id[2] + id[1] * range[2] + id[0] * range[1] * range[2];
}

/**
Computes a multi-dimensional index such that id = unlinearize(range,
linearize(range, id)) if id is a valid index in range. */
template <unsigned int dimension>
sycl::id<dimension> unlinearize(sycl::range<dimension> range, size_t id);

inline sycl::id<1> unlinearize(sycl::range<1>, size_t id) { return {id}; }

inline sycl::id<2> unlinearize(sycl::range<2> range, size_t id) {
  size_t id0 = id / range[1];
  size_t id1 = id % range[1];
  return {id0, id1};
}

inline sycl::id<3> unlinearize(sycl::range<3> range, size_t id) {
  size_t id0 = id / (range[1] * range[2]);
  size_t rem = id % (range[1] * range[2]);
  size_t id1 = rem / range[2];
  size_t id2 = rem % range[2];
  return {id0, id1, id2};
}

#endif  // __SYCLCTS_TESTS_COMMON_RANGE_INDEX_SPACE_ID_H
