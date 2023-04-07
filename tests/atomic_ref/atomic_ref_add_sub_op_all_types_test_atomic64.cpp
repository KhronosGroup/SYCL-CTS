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
//  Provides sycl::atomic_ref operator+=()/operator-=() test for atomic64 types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "atomic_ref_add_sub_op_all_types_test.h"

#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP

namespace atomic_ref::tests::api::core::atomic64 {

// FIXME: re-enable for computecpp when
// sycl::access::address_space::generic_space and possibility of a SYCL kernel
// with an unnamed type are implemented in computecpp, re-enable for hipsycl
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// hipsycl
DISABLED_FOR_TEST_CASE(ComputeCpp, hipSYCL)
("sycl::atomic_ref operator+=()/operator-=() test. atomic64 types",
 "[atomic_ref]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::atomic64)) {
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  }
  const auto type_pack = atomic_ref::tests::common::get_atomic64_types();
  for_all_types<atomic_ref::tests::api::run_add_sub_op_all_types_test>(
      type_pack);
});

DISABLED_FOR_TEST_CASE(ComputeCpp, hipSYCL)
("sycl::atomic_ref operator+=()/operator-=() test. double type",
 "[atomic_ref]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64) or
      !queue.get_device().has(sycl::aspect::atomic64)) {
    SKIP(
        "Device does not support fp64 or atomic64 operations. "
        "Skipping the test case for double type.");
  }
  const auto type_pack_fp64 = get_cts_types::get_fp64_type();
  for_all_types<atomic_ref::tests::api::run_add_sub_op_all_types_test>(
      type_pack_fp64);
});

}  // namespace atomic_ref::tests::api::core::atomic64
