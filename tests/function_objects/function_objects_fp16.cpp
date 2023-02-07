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

#include "../../util/extensions.h"
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/type_coverage.h"

// fixme: re-enable when sycl::marray is implemented in hipSYCL
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "function_objects.h"
#endif

DISABLED_FOR_TEST_CASE(ComputeCpp, hipSYCL)
("function objects void specializations scalar fp16",
 "[function_objects][fp16]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  using availability = sycl_cts::util::extensions::availability<
      sycl_cts::util::extensions::tag::fp16>;
  if (!availability::check(queue)) {
    SKIP("Device does not support half precision floating point operations.");
  }

  const auto types = named_type_pack<TYPES>::generate(TYPE_NAMES);
  const auto types_only_half = named_type_pack<sycl::half>::generate("half");

#if 0
  // prevent testing duplicate combinations
  for_all_combinations<check_scalar_return_type>(get_op_types(), types,
                                                 types_only_half);
  for_all_combinations<check_scalar_return_type>(get_op_types(),
                                                 types_only_half, types);
#else
  WARN(
      "Specification does not provide rules for the implicit conversion "
      "surrounding sycl::half. Since most types can be converted to sycl::half"
      "and vice versa, the type of the conditional operator is ambiguous."
      "See https://github.com/KhronosGroup/SYCL-Docs/issues/350");
#endif
  for_all_combinations<check_scalar_return_type>(
      get_op_types(), types_only_half, types_only_half);
});

DISABLED_FOR_TEST_CASE(DPCPP, ComputeCpp, hipSYCL)
("function objects void specializations vector fp16",
 "[function_objects][fp16]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  using availability = sycl_cts::util::extensions::availability<
      sycl_cts::util::extensions::tag::fp16>;
  if (!availability::check(queue)) {
    SKIP("Device does not support half precision floating point operations.");
  }

  const auto types_vector_only_half =
      named_type_pack<sycl::vec<sycl::half, 3>, sycl::marray<sycl::half, 5>>::
          generate("sycl::vec<sycl::half, 3>", "sycl::marray<sycl::half, 5>");

  for_all_combinations<check_vector_return_type>(get_op_types(),
                                                 types_vector_only_half);
});
