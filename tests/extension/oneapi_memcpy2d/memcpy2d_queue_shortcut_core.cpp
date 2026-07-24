/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests to check queue class functions with core types gained with
//  oneapi_memcpy2d extension
//
*******************************************************************************/
#include "memcpy2d_queue_shortcut.h"

namespace memcpy2d_queue_shortcut_core_tests {
using namespace memcpy2d_common_tests;
using namespace memcpy2d_queue_shortcut_tests;

TEST_CASE("Check queue shortcuts memcpy2d extension. core types.",
          "[oneapi_memcpy2d]") {
  auto src_ptr_types = get_pointer_types();
  auto dest_ptr_types = get_pointer_types();
  auto type_pack = get_conformance_type_pack();

  auto queue = sycl_cts::util::get_cts_object::queue();

#if !defined(SYCL_EXT_ONEAPI_MEMCPY2D)
  SKIP("SYCL_EXT_ONEAPI_MEMCPY2D is not defined");
#else
  for_all_combinations<run_queue_shortcut_tests>(type_pack, src_ptr_types,
                                                 dest_ptr_types, queue);
#endif
}

}  // namespace memcpy2d_queue_shortcut_core_tests
