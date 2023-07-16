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
//  Provides tests to check sub_group_mask none()
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "sub_group_mask_common.h"

namespace sub_group_mask_none_test {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_none_false {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    return !sub_group_mask.none();
  }
};

struct check_result_none_true {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    return sub_group_mask.none();
  }
};

struct check_type_none {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<bool, decltype(sub_group_mask.none())>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_none_false, check_type_none,
                   even_predicate, const sycl::ext::oneapi::sub_group_mask>;
template <size_t SGSize>
using verification_func_for_false_predicate =
    check_mask_api<SGSize, check_result_none_true, check_type_none,
                   false_predicate, const sycl::ext::oneapi::sub_group_mask>;
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

TEST_CASE("Check none() for mask with even predicate",
          "[oneapi_sub_group_mask]") {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  check_diff_sub_group_sizes<verification_func_for_even_predicate>();
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif
}

TEST_CASE("Check none() for mask with false predicate",
          "[oneapi_sub_group_mask]") {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  check_diff_sub_group_sizes<verification_func_for_false_predicate>();
#else
  SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif
}

}  // namespace sub_group_mask_none_test
