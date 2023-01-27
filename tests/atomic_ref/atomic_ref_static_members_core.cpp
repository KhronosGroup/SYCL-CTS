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
//  Provides sycl::atomic_ref static members test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#if !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "atomic_ref_static_members.h"

#endif  // !SYCL_CTS_COMPILING_WITH_COMPUTECPP

namespace atomic_ref::static_members::core {

// FIXME: re-enable when sycl::access::address_space::generic_space is
// implemented in computecpp
DISABLED_FOR_TEST_CASE(ComputeCpp)
("sycl::atomic_ref static members. core types", "[atomic_ref]")({
  const auto types = atomic_ref::tests::common::get_conformance_type_pack();
  for_all_types<atomic_ref::static_members::run_test>(types);
});

}  // namespace atomic_ref::static_members::core