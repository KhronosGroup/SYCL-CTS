/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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
