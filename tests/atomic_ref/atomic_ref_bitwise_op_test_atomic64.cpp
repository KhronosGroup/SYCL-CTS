/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref coperator^=()/operator|=()/operator&=()
//  test for atomic64 types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

#include "atomic_ref_bitwise_op_test.h"

#endif  // !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

namespace atomic_ref::tests::api::core::atomic64 {

// FIXME: re-enable for adaptivecpp
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// adaptivecpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref operator^=()/operator|=()/operator&=() test. atomic64 types",
 "[atomic_ref]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::atomic64)) {
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  }
  const auto type_pack = atomic_ref::tests::common::get_atomic64_types();
  for_all_types<atomic_ref::tests::api::run_bitwise_op_test>(type_pack);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref operator^=()/operator|=()/operator&=() test. double type",
 "[atomic_ref]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64) or
      !queue.get_device().has(sycl::aspect::atomic64)) {
    SKIP(
        "Device does not support fp64 or atomic64 operations. "
        "Skipping the test case for double type.");
  }
  const auto type_pack_fp64 = get_cts_types::get_fp64_type();
  for_all_types<atomic_ref::tests::api::run_bitwise_op_test>(type_pack_fp64);
});

}  // namespace atomic_ref::tests::api::core::atomic64
