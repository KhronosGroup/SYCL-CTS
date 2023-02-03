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

#if SYCL_CTS_ENABLE_HALF_TESTS && SYCL_CTS_ENABLE_DOUBLE_TESTS

#include "../../util/extensions.h"
#include "../common/common.h"
#include "../common/type_coverage.h"
#include "function_objects.h"

TEST_CASE("function objects void specializations scalar fp16 fp64",
          "[function_objects][fp16][fp64]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  using availability_fp16 = sycl_cts::util::extensions::availability<
      sycl_cts::util::extensions::tag::fp16>;
  using availability_fp64 = sycl_cts::util::extensions::availability<
      sycl_cts::util::extensions::tag::fp64>;
  if (!(availability_fp16::check(queue) && availability_fp64::check(queue))) {
    SKIP(
        "Device does not support both half and double precision floating point "
        "operations.");
  }

  const auto types_half_double =
      named_type_pack<sycl::half, double>::generate("half", "double");

#if 0
  for_all_combinations<check_scalar_return_type>(
      get_op_types(), types_half_double, types_half_double);
#else
  WARN(
      "Specification does not provide rules for the implicit conversion "
      "surrounding sycl::half. Since most types can be converted to sycl::half"
      "and vice versa, the type of the conditional operator is ambiguous.");
#endif
};

  // function objects void specializations vector fp16 fp64
  // not needed: vector type can only call operator with same operand type

#endif  // SYCL_CTS_ENABLE_HALF_TESTS && SYCL_CTS_ENABLE_DOUBLE_TESTS
