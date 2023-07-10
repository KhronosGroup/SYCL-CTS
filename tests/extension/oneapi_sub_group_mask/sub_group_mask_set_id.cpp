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
//  Provides tests to check sub_group_mask set(id)
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "sub_group_mask_common.h"

namespace sub_group_mask_set_id_test {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_set_id {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &sub_group) {
    for (size_t N = 0; N < sub_group_mask.size(); N++) {
      if (N % 3 == 0)
        sub_group_mask.set(sycl::id(N), true);
      else if (N % 3 == 1)
        sub_group_mask.set(sycl::id(N), false);
    }

    for (size_t N = 0; N < sub_group_mask.size(); N++) {
      switch (N % 3) {
        case 0:
          if (!sub_group_mask.test(sycl::id(N))) return false;
          break;
        case 1:
          if (sub_group_mask.test(sycl::id(N))) return false;
          break;
        case 2:
          if (sub_group_mask.test(sycl::id(N)) != (N % 2 == 0)) return false;
          break;
      }
    }
    return true;
  }
};

struct check_type_set_id {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<void, decltype(sub_group_mask.set(sycl::id()))>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_set_id, check_type_set_id,
                   even_predicate, sycl::ext::oneapi::sub_group_mask>;
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

TEST_CASE("Check set(id) for mask with even predicate",
          "[oneapi_sub_group_mask]") {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  check_diff_sub_group_sizes<verification_func_for_even_predicate>();
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif
}

}  // namespace sub_group_mask_set_id_test
