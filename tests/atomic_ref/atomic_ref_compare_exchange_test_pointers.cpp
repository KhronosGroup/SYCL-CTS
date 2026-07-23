/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref compare_exchange_strong()/compare_exchange_weak()
//  tests for pointers types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

#include "atomic_ref_compare_exchange_test.h"

#endif  // !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

namespace atomic_ref::tests::api::core {

// FIXME: re-enable for adaptivecpp
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// adaptivecpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref compare_exchange_strong()/compare_exchange_weak() test. "
 "pointers types",
 "[atomic_ref]")({
  const auto type_pack =
      atomic_ref::tests::common::get_conformance_pointers_type_pack();
  if (is_64_bits_pointer<void *>() && device_has_not_aspect_atomic64()) {
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  }
  for_all_types<atomic_ref::tests::api::run_compare_exchange_test>(type_pack);
});

}  // namespace atomic_ref::tests::api::core
