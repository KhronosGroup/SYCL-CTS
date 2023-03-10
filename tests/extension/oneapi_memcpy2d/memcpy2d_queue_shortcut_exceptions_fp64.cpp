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
//  Provides tests to check queue class functions exceptions with fp64 type
//  gained with oneapi_memcpy2d extension
//
*******************************************************************************/
#include "memcpy2d_queue_shortcut_exceptions.h"

namespace memcpy2d_queue_shortcut_exceptions_core_tests {
using namespace memcpy2d_common_tests;
using namespace memcpy2d_queue_shortcut_exceptions_tests;

TEST_CASE("Check queue shortcuts memcpy2d extension exceptions. fp64 type.",
          "[oneapi_memcpy2d]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
  }

  auto src_ptr_types = get_pointer_types();
  auto dest_ptr_types = get_pointer_types();
  auto type_pack = named_type_pack<double>::generate("double");

#if !defined(SYCL_EXT_ONEAPI_MEMCPY2D)
  SKIP("SYCL_EXT_ONEAPI_MEMCPY2D is not defined");
#else
  for_all_combinations<run_queue_shortcut_exceptions_tests>(
      type_pack, src_ptr_types, dest_ptr_types, queue);
#endif
}

}  // namespace memcpy2d_queue_shortcut_exceptions_core_tests
