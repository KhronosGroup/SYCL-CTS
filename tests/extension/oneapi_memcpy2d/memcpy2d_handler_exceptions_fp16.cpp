/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests to check handler class functions exceptions with fp16 type
//  gained with oneapi_memcpy2d extension
//
*******************************************************************************/
#include "memcpy2d_handler_exceptions.h"

namespace memcpy2d_handler_exceptions_fp16_tests {
using namespace memcpy2d_common_tests;
using namespace memcpy2d_handler_exceptions_tests;

TEST_CASE("Check handler memcpy2d extension exceptions. fp16 type.",
          "[oneapi_memcpy2d]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
  }

  auto src_ptr_types = get_pointer_types();
  auto dest_ptr_types = get_pointer_types();
  auto type_pack = named_type_pack<sycl::half>::generate("sycl::half");

#if !defined(SYCL_EXT_ONEAPI_MEMCPY2D)
  SKIP("SYCL_EXT_ONEAPI_MEMCPY2D is not defined");
#else
  for_all_combinations<run_handler_exceptions_tests>(type_pack, src_ptr_types,
                                                     dest_ptr_types, queue);
#endif
}

}  // namespace memcpy2d_handler_exceptions_fp16_tests
