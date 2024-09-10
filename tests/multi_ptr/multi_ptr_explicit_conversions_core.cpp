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

//  Provides tests for multi_ptr explicit conversions for core types

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_explicit_conversions.h"

namespace multi_ptr_explicit_conversions_core {

TEST_CASE("multi_ptr explicit conversions. Core types", "[multi_ptr]") {
  using namespace multi_ptr_explicit_conversions;
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_multi_ptr_explicit_convert_for_type>(types);
  for_all_types<check_multi_ptr_explicit_convert_for_type>(composite_types);
}

DISABLED_FOR_TEST_CASE(DPCPP, hipSYCL)
("generic_ptr alias. Core types", "[multi_ptr]")({
  using namespace multi_ptr_explicit_conversions;
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_generic_ptr_aliases_for_type>(types);
  for_all_types<check_generic_ptr_aliases_for_type>(composite_types);
});
}  // namespace multi_ptr_explicit_conversions_core

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
