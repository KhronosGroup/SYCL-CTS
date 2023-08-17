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
//  Provides tests to check sub_group_mask extract_bits()
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "sub_group_mask_common.h"

namespace sub_group_mask_extract_bits_test {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

// Since sub_group_mask with even predicate consists of 0101...01
// expected extracted bits are 0101..01 or 1010..10 depending on starting pos.
// Filling variable of type T with 01 or 10 to match size of the mask
// and the rest of starting bits are remaining 0.
template <typename T>
void get_expected_bits(T &out, uint32_t mask_size, int pos) {
  if (pos >= mask_size - 1) return;
  int init;
  if (pos % 2 == 0)
    init = 0b01;
  else
    init = 0b10;
  out = init;
  for (size_t i = 2; i + 2 <= sizeof(T) * CHAR_BIT && i + 2 <= mask_size - pos;
       i = i + 2) {
    out <<= 2;
    out += init;
  }
}

// For marray the extracted bits start from the first element and moves position
// into the consecutive elements.
template <typename T, size_t N>
void get_expected_bits(sycl::marray<T, N>& out, uint32_t mask_size, int pos) {
  for (size_t i = 0; i < N; ++i)
    get_expected_bits(out[i], mask_size, pos + i * sizeof(T) * CHAR_BIT);
}

template <typename T>
struct check_result_extract_bits {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group&) {
    for (size_t pos = 0; pos <= sub_group_mask.size(); pos++) {
      T bits;
      sub_group_mask.extract_bits(bits, sycl::id(pos));
      T expected = value_operations::init<T>(0);
      get_expected_bits(expected, sub_group_mask.size(), pos);
      if (!value_operations::are_equal(bits, expected)) return false;
    }
    return true;
  }
};

template <typename T>
struct check_type_extract_bits {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    T bits;
    return std::is_same_v<void, decltype(sub_group_mask.extract_bits(bits))>;
  }
};

template <typename T>
struct check_for_type {
  template <size_t SGSize>
  using verification_func_for_even_predicate =
      check_mask_api<SGSize, check_result_extract_bits<T>,
                     check_type_extract_bits<T>, even_predicate,
                     const sycl::ext::oneapi::sub_group_mask>;

  void operator()(const std::string& typeName) {
    SECTION("testing: " + type_name_string<T>::get(typeName)) {
      check_diff_sub_group_sizes<verification_func_for_even_predicate>();
    }
  }
};
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

TEST_CASE("Check extract_bits() for mask with even predicate",
          "[oneapi_sub_group_mask]") {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  for_all_types_and_marrays<check_for_type>(types);
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif
}

}  // namespace sub_group_mask_extract_bits_test
