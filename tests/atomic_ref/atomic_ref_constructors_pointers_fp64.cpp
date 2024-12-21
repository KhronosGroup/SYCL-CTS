/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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
//  Provides sycl::atomic_ref constructors test for double *.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

#include "atomic_ref_constructors.h"

#endif  // !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

namespace atomic_ref::tests::constructors::core {

// FIXME: re-enable for adaptivecpp
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// adaptivecpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("sycl::atomic_ref constructors. double *", "[atomic_ref]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP(
        "Device does not support double precision floating point "
        "operations.");
  }
  const auto types = atomic_ref::tests::common::get_fp64_pointers_type_pack();
  if (is_64_bits_pointer<void*>() && device_has_not_aspect_atomic64()) {
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
  }
  for_all_types<atomic_ref::tests::constructors::run_test>(types);
});

}  // namespace atomic_ref::tests::constructors::core
