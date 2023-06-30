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
//  Provides tests to check sub_group_mask insert_bits()
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "sub_group_mask_common.h"

namespace sub_group_mask_insert_bits_test {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

// to get 0b0101.. to insert
template <typename T>
void get_bits(T &out) {
  out = 1;
  for (size_t i = 2; i + 2 <= sizeof(T) * CHAR_BIT; i = i + 2) {
    out <<= 2;
    out++;
  }
}

template <typename T, size_t nElements>
void get_bits(sycl::marray<T, nElements> &out) {
  T val;
  get_bits(val);
  std::fill(out.begin(), out.end(), val);
}

template <typename T>
struct check_result_insert_bits {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    for (size_t pos = 0; pos < sub_group_mask.size(); pos++) {
      sycl::ext::oneapi::sub_group_mask mask = sub_group_mask;
      T bits;
      get_bits(bits);
      mask.insert_bits(bits, sycl::id(pos));
      for (size_t K = 0; K < mask.size(); K++)
        if (K >= pos && K < pos + CHAR_BIT * sizeof(T)) {
          if (mask.test(sycl::id(K)) != ((K - pos) % 2 == 0)) return false;
        } else {
          if (mask.test(sycl::id(K)) != (K % 3 == 0)) return false;
        }
    }
    return true;
  }
};

template <typename T>
struct check_type_insert_bits {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same_v<void, decltype(sub_group_mask.insert_bits(T()))>;
  }
};

template <typename T>
struct check_for_type {
  template <size_t SGSize>
  using verification_func_for_mod3_predicate =
      check_mask_api<SGSize, check_result_insert_bits<T>,
                     check_type_insert_bits<T>, mod3_predicate,
                     sycl::ext::oneapi::sub_group_mask>;

  void operator()(const std::string &typeName) {
    SECTION("testing: " + type_name_string<T>::get(typeName)) {
      check_diff_sub_group_sizes<verification_func_for_mod3_predicate>();
    }
  }
};
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

TEST_CASE("Check insert_bits() for mask with mod3 predicate",
          "[oneapi_sub_group_mask]") {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  for_all_types_and_marrays<check_for_type>(types);
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif
}

}  // namespace sub_group_mask_insert_bits_test
