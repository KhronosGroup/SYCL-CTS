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
//  Provides sycl::atomic_ref compare_exchange_strong()/compare_exchange_weak()
//  tests for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "atomic_ref_compare_exchange_test.h"

#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL

namespace atomic_ref::tests::api::core {

// FIXME: re-enable for hipsycl
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// hipsycl
DISABLED_FOR_TEST_CASE(hipSYCL)
("sycl::atomic_rer compare_exchange_strong()/compare_exchange_weak() test. "
 "core types",
 "[atomic_ref]")({
  const auto type_pack = atomic_ref::tests::common::get_conformance_type_pack();
  for_all_types<atomic_ref::tests::api::run_compare_exchange_test>(type_pack);
});

}  // namespace atomic_ref::tests::api::core
