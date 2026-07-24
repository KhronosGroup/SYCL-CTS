/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref static members test for atomic64 types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "atomic_ref_static_members.h"

namespace atomic_ref::tests::static_members::core::atomic64 {

TEST_CASE("sycl::atomic_ref static members. atomic64 types", "[atomic_ref]") {
  const auto type_pack = atomic_ref::tests::common::get_atomic64_types();
  for_all_types<atomic_ref::tests::static_members::run_test>(type_pack);

  const auto type_pack_fp64 = get_cts_types::get_fp64_type();
  for_all_types<atomic_ref::tests::static_members::run_test>(type_pack_fp64);
}

}  // namespace atomic_ref::tests::static_members::core::atomic64
