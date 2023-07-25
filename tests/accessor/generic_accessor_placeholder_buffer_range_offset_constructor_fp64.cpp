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
//  Provides generic sycl::accessor placeholder buffer range offset constructor
//  test for double type
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "accessor_common.h"
#include "generic_accessor_placeholder_buffer_range_offset_constructor.h"

using namespace generic_accessor_placeholder_buffer_range_offset_constructor;
#endif

#include "../common/disabled_for_test_case.h"

namespace generic_accessor_placeholder_buffer_range_offset_constructor_fp64 {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL, ComputeCpp)
("Generic sycl::accessor placeholder buffer range offset constructor. fp64 "
 "type",
 "[accessor]", test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
    for_type_vectors_marray<
        run_generic_placeholder_buffer_range_offset_constructor_test, double,
        TestType>("double");
#else
    run_generic_placeholder_buffer_range_offset_constructor_test<double,
                                                                 TestType>{}(
        "double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  } else {
    WARN("Device does not support double precision floating point operations");
    return;
  }
});

}  // namespace
   // generic_accessor_placeholder_buffer_range_offset_constructor_fp64
