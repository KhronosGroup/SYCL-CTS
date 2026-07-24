/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

//  Check that constant evaluated code inside a kernel with some usually
//  forbidden constructs works.

#include "../common/get_cts_object.h"
#include "catch2/catch_test_macros.hpp"

namespace language::constant_evaluation {

// A very inefficient way to compute Fibonacci's series using recursion which is
// forbidden inside non constant-evaluated kernel code.
auto constexpr f(int v) {
  if (v < 2) return v;
  return f(v - 1) + f(v - 2);
}

// This corresponds to clarification introduced with
// https://github.com/KhronosGroup/SYCL-Docs/pull/388
TEST_CASE("constant evaluated code works inside kernel", "[language]") {
  sycl::buffer<int> b{1};
  sycl_cts::util::get_cts_object::queue().submit([&](sycl::handler& cgh) {
    sycl::accessor a{b, cgh, sycl::write_only};
    cgh.single_task([=] {
      // A constant-evaluated recursive function.
      constexpr auto result = f(10);
      a[0] = result;

      if constexpr (false) {
        // A non constant-evaluated recursive function going further more
        // through forbidden function pointer which is skipped by the if
        // constexpr.
        auto* p = f;
        auto other = p(5) + f(6);
      }
    });
  });
  CHECK(sycl::host_accessor{b}[0] == 55);
}

}  // namespace language::constant_evaluation
