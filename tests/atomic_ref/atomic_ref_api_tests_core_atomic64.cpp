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
//  Provides sycl::atomic_ref api test for atomic64 types
//
*******************************************************************************/
#include "atomic_ref_api_tests.h"

namespace atomic_ref::tests::api::core::atomic64 {

TEST_CASE("sycl::atomic_ref api tests. atomic64 types", "[atomic_ref]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::atomic64)) {
    SKIP(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
    return;
  }
  const auto type_pack = atomic_ref::tests::common::get_atomic64_types();
  for_all_types<atomic_ref::tests::api::run_tests>(type_pack);
}

}  // namespace atomic_ref::tests::api::core::atomic64
