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
*******************************************************************************/

//  Check that constant evaluated code inside a kernel with some usually
//  forbidden constructs works.

#include "../common/get_cts_object.h"
#include "catch2/catch_test_macros.hpp"

namespace language::constant_evaluation {

// A very inefficient way to compute Fibonacci's series using recursion which is
// forbidden inside non constant-evaluated kernel code/
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
