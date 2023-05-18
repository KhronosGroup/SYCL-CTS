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
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_WEAK_OBJECT_COMMON_H
#define __SYCLCTS_TESTS_WEAK_OBJECT_COMMON_H

#include "../../../util/sycl_exceptions.h"
#include "../../common/common.h"
#include "../../common/type_coverage.h"

#ifdef SYCL_EXT_ONEAPI_WEAK_OBJECT

namespace weak_object_common {

template <typename SYCLObjT>
SYCLObjT get_sycl_object() {
  static sycl::buffer<int> buf{{1}};
  static sycl::buffer<int> host_buf{{1}};

  if constexpr (std::is_same_v<SYCLObjT, sycl::buffer<int>>) {
    return {sycl::range{1}};
  } else if constexpr (std::is_same_v<SYCLObjT, sycl::accessor<int>>) {
    return {buf};
  } else if constexpr (std::is_same_v<SYCLObjT, sycl::host_accessor<int>>) {
    return {host_buf};
  } else {
    return SYCLObjT{};
  }
}
}  // namespace weak_object_common
#endif  // SYCL_EXT_ONEAPI_WEAK_OBJECT

#endif  // __SYCLCTS_TESTS_WEAK_OBJECT_COMMON_H
