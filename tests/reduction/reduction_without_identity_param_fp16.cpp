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
//  Provides tests for interaction reductions with sycl::half variable type
//  without identity param.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "reduction_without_identity_param_common.h"
#endif

namespace reduction_without_identity_param_fp16 {

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_without_identity_param_fp16", "[reduction]")({
  using namespace reduction_without_identity_param_common;

  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  run_tests_for_all_functors<sycl::half, run_test_without_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors<sycl::half, run_test_without_property>()(
      reduction_common::nd_range, queue, "sycl::half");
  run_tests_for_all_functors<sycl::half, run_test_with_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors<sycl::half, run_test_with_property>()(
      reduction_common::nd_range, queue, "sycl::half");
});

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_without_identity_param_item_twice_fp16", "[reduction]")({
  using namespace reduction_without_identity_param_common;

  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  run_tests_for_all_functors_item_twice<sycl::half,
                                        run_test_without_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors_item_twice<sycl::half,
                                        run_test_without_property>()(
      reduction_common::nd_range, queue, "sycl::half");
  run_tests_for_all_functors_item_twice<sycl::half, run_test_with_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors_item_twice<sycl::half, run_test_with_property>()(
      reduction_common::nd_range, queue, "sycl::half");
});

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_without_identity_param_even_item_fp16", "[reduction]")({
  using namespace reduction_without_identity_param_common;

  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  run_tests_for_all_functors_even_item<sycl::half, run_test_without_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors_even_item<sycl::half, run_test_without_property>()(
      reduction_common::nd_range, queue, "sycl::half");
  run_tests_for_all_functors_even_item<sycl::half, run_test_with_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors_even_item<sycl::half, run_test_with_property>()(
      reduction_common::nd_range, queue, "sycl::half");
});

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reduction_without_identity_param_no_one_item_fp16", "[reduction]")({
  using namespace reduction_without_identity_param_common;

  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  run_tests_for_all_functors_no_one_item<sycl::half,
                                         run_test_without_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors_no_one_item<sycl::half,
                                         run_test_without_property>()(
      reduction_common::nd_range, queue, "sycl::half");
  run_tests_for_all_functors_no_one_item<sycl::half, run_test_with_property>()(
      reduction_common::range, queue, "sycl::half");
  run_tests_for_all_functors_no_one_item<sycl::half, run_test_with_property>()(
      reduction_common::nd_range, queue, "sycl::half");
});
}  // namespace reduction_without_identity_param_fp16
