/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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
//  This file is a common header for implementing buffer, local, and image
//  accessor tests
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_COMMON_ALL_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_COMMON_ALL_H

#include "../common/common.h"
#include "accessor_utility.h"

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

namespace {

/** tests accessor member values
*/
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
void check_accessor_members(sycl_cts::util::logger &log) {
#ifdef VERBOSE_LOG
  log_accessor<T, dims, mode, target, placeholder>("check_accessor_members",
                                                   log);
#endif  // VERBOSE_LOG

  using acc_t = cl::sycl::accessor<T, dims, mode, target, placeholder>;

  using value_type = typename acc_t::value_type;
  static_assert(std::is_same<value_type, T>::value,
                "value_type is of wrong type");

  using reference = typename acc_t::reference;
  static_assert(std::is_same<reference, T &>::value,
                "reference is of wrong type");

  using const_reference = typename acc_t::const_reference;
  static_assert(std::is_same<const_reference, const T &>::value,
                "const_reference is of wrong type");
}

}  // namespace

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_COMMON_ALL_H
