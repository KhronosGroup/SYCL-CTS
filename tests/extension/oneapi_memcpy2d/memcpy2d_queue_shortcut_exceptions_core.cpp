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
//  Provides tests to check queue class functions exceptions with core types
//  gained with oneapi_memcpy2d extension
//
*******************************************************************************/
#include "memcpy2d_queue_shortcut_exceptions.h"

namespace memcpy2d_queue_shortcut_exceptions_core_tests {
using namespace memcpy2d_common_tests;
using namespace memcpy2d_queue_shortcut_exceptions_tests;

TEST_CASE("Check queue shortcuts memcpy2d extension exceptions. core types.",
          "[oneapi_memcpy2d]") {
  auto src_ptr_types = get_pointer_types();
  auto dest_ptr_types = get_pointer_types();
  auto type_pack = get_conformance_type_pack();

  auto queue = sycl_cts::util::get_cts_object::queue();

#if !defined(SYCL_EXT_ONEAPI_MEMCPY2D)
  SKIP("SYCL_EXT_ONEAPI_MEMCPY2D is not defined");
#else
  for_all_combinations<run_queue_shortcut_exceptions_tests>(
      type_pack, src_ptr_types, dest_ptr_types, queue);
#endif
}

}  // namespace memcpy2d_queue_shortcut_exceptions_core_tests
