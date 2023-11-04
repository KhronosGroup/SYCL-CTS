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
//  Provides tests to check sub_group_mask constructors()
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "sub_group_mask_common.h"

namespace sub_group_mask_constructors_test {

using namespace sycl_cts;

#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
template <typename T, size_t dim>
void copy_ulong_long2marray(sycl::marray<T, dim>& marray,
                            unsigned long long value) {
  size_t elem_size = sizeof(T);
  size_t num_bytes =
      elem_size * dim > sizeof(value) ? sizeof(value) : elem_size * dim;
  size_t shift = 0;
  size_t i = 0;
  auto valuePtr = reinterpret_cast<char*>(&value);
  while (shift + elem_size <= num_bytes && i < dim) {
    memcpy(&(marray[i]), valuePtr + shift, elem_size);
    ++i;
    shift += elem_size;
  }
}

template <typename T, size_t dim>
void copy_marray2ulong_long(unsigned long long& value,
                            const sycl::marray<T, dim>& marray) {
  size_t elem_size = sizeof(T);
  size_t num_bytes =
      elem_size * dim > sizeof(value) ? sizeof(value) : elem_size * dim;
  size_t shift = 0;
  size_t i = 0;
  auto valuePtr = reinterpret_cast<char*>(&value);
  while (shift + elem_size <= num_bytes && i < dim) {
    memcpy(valuePtr + shift, &(marray[i]), elem_size);
    ++i;
    shift += elem_size;
  }
}

bool compare_mask_and_init_value(sycl::ext::oneapi::sub_group_mask& mask,
                                 unsigned long long init_value) {
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[sycl::id<1>(i)] != (((init_value >> i) & 1))) {
      return false;
    }
  }
  return true;
}

template <typename T, size_t dim>
bool compare_mask_and_init_value(sycl::ext::oneapi::sub_group_mask& mask,
                                 const sycl::marray<T, dim>& val) {
  unsigned long long result = 0;
  size_t num_bytes = (sizeof(val) >= mask.size()) ? mask.size() : sizeof(val);
  copy_marray2ulong_long(result, val);
  return compare_mask_and_init_value(mask, result);
}

unsigned long long init_even_bits() {
  unsigned long long value = 0;
  for (size_t i = 0; i < CHAR_BIT * sizeof(value); i += 2) {
    value |= 1UL << i;
  }
  return value;
}

template <typename T>
res_array check_mask_constructors(sycl::ext::oneapi::sub_group_mask& mask,
                                  T& value) {
  res_array result;
  result[ctor_error::ctor_wrong] = compare_mask_and_init_value(mask, value);
  sycl::ext::oneapi::sub_group_mask copy(mask);
  result[ctor_error::copy_ctor_wrong] =
      compare_mask_and_init_value(copy, value);
  auto assign = (copy = mask);
  result[ctor_error::assign_type_wrong] =
      std::is_same_v<decltype(assign), decltype(copy)>;
  result[ctor_error::assign_wrong] = compare_mask_and_init_value(assign, value);
  return result;
}

struct check_result_default_ctor {
  res_array operator()(sycl::nd_item<1>& nd_item) {
    sycl::ext::oneapi::sub_group_mask mask;
    unsigned long long value = 0;
    return check_mask_constructors(mask, value);
  }
};

struct check_result_ctor_ulong_long {
  res_array operator()(sycl::nd_item<1>& item) {
    // Check for value = item.get_local_linear_id()
    unsigned long long val = item.get_local_linear_id();
    sycl::ext::oneapi::sub_group_mask mask(val);
    res_array result = check_mask_constructors(mask, val);
    // Check for value = 0b1010...
    val = init_even_bits();
    sycl::ext::oneapi::sub_group_mask mask_even(val);
    result |= check_mask_constructors(mask_even, val);
    return result;
  }
};

template <typename T>
struct check_result_ctor_marray {
  res_array operator()(sycl::nd_item<1>& item) {
    // Check for value = item.get_local_linear_id()
    unsigned long long value = item.get_local_linear_id();
    T val{};
    copy_ulong_long2marray(val, value);
    sycl::ext::oneapi::sub_group_mask mask(val);
    res_array result = check_mask_constructors(mask, val);
    // Check for value = 0b1010...
    value = init_even_bits();
    val = T{};
    copy_ulong_long2marray(val, value);
    sycl::ext::oneapi::sub_group_mask mask_even(val);
    result |= check_mask_constructors(mask_even, val);
    return result;
  }
};

template <size_t SGSize>
using verification_func_for_def_ctor =
    check_mask_ctors<SGSize, check_result_default_ctor>;

template <size_t SGSize>
using verification_func_for_ulong_long_ctor =
    check_mask_ctors<SGSize, check_result_ctor_ulong_long>;

template <typename T>
struct check_for_type {
  template <size_t SGSize>
  using verification_func_for_marray_ctor =
      check_mask_ctors<SGSize, check_result_ctor_marray<T>>;

  void operator()(const std::string& typeName) {
    check_diff_sub_group_sizes<verification_func_for_marray_ctor>();
  }
};

#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2

TEST_CASE("Check sub_group_mask empty constructor", "[oneapi_sub_group_mask]") {
#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
  check_diff_sub_group_sizes<verification_func_for_def_ctor>();
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined or revision less than 2");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
};

TEST_CASE("Check sub_group_mask(unsigned long long) constructor",
          "[oneapi_sub_group_mask]") {
#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
  check_diff_sub_group_sizes<verification_func_for_ulong_long_ctor>();
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined or revision less than 2");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
};

TEST_CASE("Check sub_group_mask(marray<T,dim>) constructor",
          "[oneapi_sub_group_mask]") {
#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
  for_marrays_of_all_types<check_for_type>(types);
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined or revision less than 2");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
};

}  // namespace sub_group_mask_constructors_test
