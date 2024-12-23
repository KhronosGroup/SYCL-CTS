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

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/type_coverage.h"

// fixme: re-enable when sycl::marray is implemented in AdaptiveCpp
#ifndef SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "function_objects.h"
#endif

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("function objects void specializations scalar core", "[function_objects]")({
  const auto types = named_type_pack<TYPES>::generate(TYPE_NAMES);
  for_all_combinations<check_scalar_return_type>(get_op_types(), types, types);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("function objects void specializations vector core", "[function_objects]")({
  const auto types_vector =
      named_type_pack<TYPES_VECTOR>::generate(TYPE_NAMES_VECTOR);
  for_all_combinations<check_vector_return_type>(get_op_types(), types_vector);
});
