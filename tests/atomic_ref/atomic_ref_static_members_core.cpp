/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref static members test for generic types
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "atomic_ref_static_members.h"

namespace atomic_ref::tests::static_members::core {

TEST_CASE("sycl::atomic_ref static members. core types", "[atomic_ref]") {
  const auto types = atomic_ref::tests::common::get_conformance_type_pack();
  for_all_types<atomic_ref::tests::static_members::run_test>(types);
}

}  // namespace atomic_ref::tests::static_members::core
