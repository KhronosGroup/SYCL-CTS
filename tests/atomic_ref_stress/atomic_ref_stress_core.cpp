/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "atomic_ref_stress_common.h"
#endif  // !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include <catch2/catch_test_macros.hpp>

namespace atomic_ref_stress_test_core {

// FIXME: re-enable for adaptivecpp
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// adaptivecpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity for device scope. core types",
 "[atomic_ref_stress]")({
  const auto type_pack = named_type_pack<int, float>::generate("int", "float");
  for_all_types<atomic_ref_stress_test::run_atomicity_device_scope>(type_pack);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity for work_group scope. core types",
 "[atomic_ref_stress]")({
  const auto type_pack = named_type_pack<int, float>::generate("int", "float");
  for_all_types<atomic_ref_stress_test::run_atomicity_work_group_scope>(
      type_pack);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref aquire and release. core types", "[atomic_ref_stress]")({
  const auto type_pack = named_type_pack<int, float>::generate("int", "float");
  for_all_types<atomic_ref_stress_test::run_aquire_release>(type_pack);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref ordering. core types", "[atomic_ref_stress]")({
  const auto type_pack = named_type_pack<int, float>::generate("int", "float");
  for_all_types<atomic_ref_stress_test::run_ordering>(type_pack);
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity with respect to atomic operations in host code. "
 "core types",
 "[atomic_ref_stress]")({
#ifdef __cpp_lib_atomic_ref
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::usm_atomic_shared_allocations))
    SKIP(
        "Device does not support usm_atomic_shared_allocations. "
        "Skipping the test case.");
  const auto type_pack = named_type_pack<int, float>::generate("int", "float");
  for_all_types<atomic_ref_stress_test::run_atomicity_with_host_code>(
      type_pack);
#else
  SKIP("std::atomic_ref is not available");
#endif
});

}  // namespace atomic_ref_stress_test_core
