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
//  Provides tests for reductions with identity parameter for sycl::half.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "reduction_with_identity_param.h"
#endif

namespace reduction_with_identity_param_fp16 {

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
("reduction_with_identity_param_fp16", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  reduction_with_identity_param::run_test_for_type<sycl::half>()(queue,
                                                                 "sycl::half");
});

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
("reduction_with_identity_param_item_twice_fp16", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  reduction_with_identity_param::run_test_for_type_item_twice<sycl::half>()(
      queue, "sycl::half");
});

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
("reduction_with_identity_param_even_item_fp16", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  reduction_with_identity_param::run_test_for_type_even_item<sycl::half>()(
      queue, "sycl::half");
});

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
("reduction_with_identity_param_no_one_item_fp16", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  reduction_with_identity_param::run_test_for_type_no_one_item<sycl::half>()(
      queue, "sycl::half");
});
}  // namespace reduction_with_identity_param_fp16
