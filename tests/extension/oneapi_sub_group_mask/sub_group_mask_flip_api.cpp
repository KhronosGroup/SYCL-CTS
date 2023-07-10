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
//  Provides tests to check sub_group_mask flip()
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "sub_group_mask_common.h"

namespace sub_group_mask_flip_api_test {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_flip {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &sub_group) {
    // sub_group_mask's size must be in the range between 0 (excluded) and 32
    // (included) to rule out UB
    if (sub_group_mask.size() > 32 || sub_group_mask.size() == 0) return false;
    unsigned long before_flip, after_flip;
    sub_group_mask.extract_bits(before_flip);
    sub_group_mask.flip();
    sub_group_mask.extract_bits(after_flip);
    // mask off irrelevant bits
    unsigned long mask =
        ULONG_MAX >> (CHAR_BIT * sizeof(unsigned long) - sub_group_mask.size());
    return ((after_flip & before_flip) & mask) == 0;
  }
};

struct check_type_flip {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<void, decltype(sub_group_mask.flip())>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_flip, check_type_flip, even_predicate,
                   sycl::ext::oneapi::sub_group_mask>;
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

TEST_CASE("Check flip() for mask with even predicate",
          "[oneapi_sub_group_mask]") {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  check_diff_sub_group_sizes<verification_func_for_even_predicate>();
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif
}

}  // namespace sub_group_mask_flip_api_test
