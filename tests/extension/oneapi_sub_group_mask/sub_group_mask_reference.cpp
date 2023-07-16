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
//  Provides tests to check sub_group_mask operator[] const and reference
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "sub_group_mask_common.h"

namespace sub_group_mask_reference_test {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_reference {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    for (size_t N = 0; N < sub_group_mask.size(); N++) {
      sycl::ext::oneapi::sub_group_mask::reference ref_to_bit =
          sub_group_mask[sycl::id(N)];
      // check that reference to bit have correct value
      if (ref_to_bit != (N % 2 == 0)) return false;
      switch (N % 5) {
        case 0:
          // check reference operator=(bool x)
          // by assigning opposite value and checking corresponding bit in mask
          ref_to_bit = (N % 2 != 0);
          if (sub_group_mask[sycl::id(N)] != (N % 2 != 0)) return false;
          break;
        case 1:
          // check reference operator=(const reference& x)
          // by assigning reference for next bit and checking corresponding bit in mask
          if (N == sub_group_mask.size() - 1) break;
          ref_to_bit = sub_group_mask[sycl::id(N + 1)];
          if (sub_group_mask[sycl::id(N)] != ((N + 1) % 2 == 0)) return false;
          break;
        case 2:
          // check reference operator~()
          if (~ref_to_bit != (N % 2 != 0)) return false;
          break;
        case 3:
          // check reference operator bool()
          if (!!ref_to_bit != (N % 2 == 0)) return false;
          break;
        case 4:
          // check reference member function flip()
          if (!std::is_same<sycl::ext::oneapi::sub_group_mask::reference &,
                            decltype(ref_to_bit.flip())>::value)
            return false;
          if (ref_to_bit.flip() != (N % 2 != 0)) return false;
          break;
      }
    }
    return true;
  }
};

struct check_type_reference {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<sycl::ext::oneapi::sub_group_mask::reference,
                        decltype(sub_group_mask[sycl::id()])>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_reference, check_type_reference,
                   even_predicate, sycl::ext::oneapi::sub_group_mask>;
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

TEST_CASE("Check operator[] and reference for mask with even predicate",
          "[oneapi_sub_group_mask]") {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  check_diff_sub_group_sizes<verification_func_for_even_predicate>();
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif
}

}  // namespace sub_group_mask_reference_test
