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
//  Provides generic sycl::accessor constructors with buffers test for generic
//  types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "accessor_common.h"
#include "generic_accessor_common_buffer_constructors.h"

using namespace generic_accessor_common_buffer_constructors;
#endif

#include "../common/disabled_for_test_case.h"

namespace generic_accessor_common_buffer_constructors_core {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Generic sycl::accessor buffer constructors. core types", "[accessor]",
 test_combinations)({
  common_run_tests<run_generic_common_buffer_constructors_test, TestType>();
});

}  // namespace generic_accessor_common_buffer_constructors_core
