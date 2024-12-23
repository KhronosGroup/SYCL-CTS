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
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "atomic_ref_stress_common.h"
#endif  // !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include <catch2/catch_test_macros.hpp>

namespace atomic_ref_stress_test_atomic64 {

// FIXME: re-enable for adaptivecpp
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// adaptivecpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity for device scope test. long long type",
 "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_atomicity_device_scope<long long>{}("long long");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity for device scope test. double type",
 "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  if (!queue.get_device().has(sycl::aspect::fp64))
    SKIP(
        "Device does not support fp64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_atomicity_device_scope<double>{}("double");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity for work_group scope test. long long type",
 "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_atomicity_work_group_scope<long long>{}(
      "long long");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity for work_group scope test. double type",
 "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  if (!queue.get_device().has(sycl::aspect::fp64))
    SKIP(
        "Device does not support fp64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_atomicity_work_group_scope<double>{}("double");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref aquire and release. long long type", "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_aquire_release<long long>{}("long long");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref aquire and release. double types", "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  if (!queue.get_device().has(sycl::aspect::fp64))
    SKIP(
        "Device does not support fp64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_aquire_release<double>{}("double");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref ordering. long long type", "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_ordering<long long>{}("long long");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref ordering. double type", "[atomic_ref_stress]")({
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  if (!queue.get_device().has(sycl::aspect::fp64))
    SKIP(
        "Device does not support fp64 operations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_ordering<double>{}("double");
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity with respect to atomic operations in host code. "
 "long long type",
 "[atomic_ref_stress]")({
#ifdef __cpp_lib_atomic_ref
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  if (!queue.get_device().has(sycl::aspect::usm_atomic_shared_allocations))
    SKIP(
        "Device does not support usm_atomic_shared_allocations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_atomicity_with_host_code<long long>{}(
      "long long");
#else
  SKIP("std::atomic_ref is not available");
#endif
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref atomicity with respect to atomic operations in host code. "
 "double type",
 "[atomic_ref_stress]")({
#ifdef __cpp_lib_atomic_ref
  auto queue = once_per_unit::get_queue();
  if (!queue.get_device().has(sycl::aspect::atomic64))
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  if (!queue.get_device().has(sycl::aspect::fp64))
    SKIP(
        "Device does not support fp64 operations. "
        "Skipping the test case.");
  if (!queue.get_device().has(sycl::aspect::usm_atomic_shared_allocations))
    SKIP(
        "Device does not support usm_atomic_shared_allocations. "
        "Skipping the test case.");

  atomic_ref_stress_test::run_atomicity_with_host_code<double>{}("double");
#else
  SKIP("std::atomic_ref is not available");
#endif
});

}  // namespace atomic_ref_stress_test_atomic64
