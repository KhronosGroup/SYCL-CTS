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

#include <sycl/sycl.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../../common/get_cts_object.h"

#include <algorithm>
#include <vector>

namespace composite_device::tests {

TEST_CASE("Test for impact on descendent device", "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "implementation");
#else

  auto component_device = sycl_cts::util::get_cts_object::device();
  if (!component_device.has(sycl::aspect::ext_oneapi_is_component)) {
    SKIP(
        "Selected device is not a component device, this test has nothing to "
        "do");
  }

  auto composite_device = component_device.get_info<
      sycl::ext::oneapi::experimental::info::device::composite_device>();

  INFO(
      "Checking that a queue for a component device can be created using "
      "context of the corresponding composite device");

  sycl::context composite_context(composite_device);
  sycl::queue q(composite_context, component_device);

  INFO(
      "Checking that a queue for a component device created using context of "
      "the corresponding composite device can be used to perform simple "
      "operations");

  constexpr size_t count = 100;
  std::vector<int> a(count), b(count);
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 42);

  sycl::buffer bufA(a);
  sycl::buffer bufB(b);
  sycl::buffer<int> bufC(sycl::range{count});

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor accA(bufA, cgh, sycl::read_only);
    sycl::accessor accB(bufB, cgh, sycl::read_only);
    sycl::accessor accC(bufC, cgh, sycl::write_only);

    cgh.parallel_for(sycl::range{count},
                     [=](sycl::id<1> it) { accC[it] = accA[it] + accB[it]; });
  });

  auto hostAcc = bufC.get_host_access();
  INFO("Verifying kernel (vector add) results");
  for (size_t i = 0; i < count; ++i) {
    REQUIRE(a[i] + b[i] == hostAcc[i]);
  }

#endif
}

}  // namespace composite_device::tests
