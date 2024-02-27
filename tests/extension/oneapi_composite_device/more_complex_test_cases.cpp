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

TEST_CASE("Basic test for a composite device", "[oneapi_composite_device]") {
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

  INFO("Checking that a queue can be created for a composite device");

  sycl::queue q(composite_device);

  INFO("Checking that a composite device queue can perform basic operations");

  std::vector<int> a(100), b(100);
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 42);

  sycl::buffer bufA(a);
  sycl::buffer bufB(b);
  sycl::buffer<int> bufC(sycl::range{100});

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor accA(bufA, cgh, sycl::read_only);
    sycl::accessor accB(bufB, cgh, sycl::read_only);
    sycl::accessor accC(bufB, cgh, sycl::write_only);

    cgh.parallel_for(sycl::range{100},
                     [=](sycl::id<1> it) { accC[it] = accA[it] + accB[it]; });
  });

  auto hostAcc = bufC.get_host_access();
  INFO("Verifying kernel (vector add) results");
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i] + b[i] == hostAcc[i]);
  }

#endif
}

TEST_CASE("Interoperability between composite and component devices",
          "[oneapi_composite_device]") {
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

  if (!component_device.has(sycl::aspect::usm_device_allocations) ||
      !composite_device.has(sycl::aspect::usm_device_allocations)) {
    SKIP(
        "Either composite or component device does not support USM device "
        "allocations, this test has nothing to do");
  }

  INFO(
      "Checking that a shared context for both composite and component device "
      "can be created");
  sycl::context shared_context({composite_device, component_device});

  sycl::queue composite_queue(shared_context, composite_device);
  sycl::queue component_queue(shared_context, component_device);

  constexpr size_t count = 100;
  auto* ptrA =
      sycl::malloc_device<int>(count, component_device, shared_context);
  auto* ptrB =
      sycl::malloc_device<int>(count, composite_device, shared_context);

  auto eventA = component_queue.parallel_for(
      sycl::range{count}, [=](sycl::id<1> it) { ptrA[it] = it; });

  auto eventB = composite_queue.parallel_for(
      sycl::range{count}, [=](sycl::id<1> it) { ptrB[it] = it; });

  sycl::buffer<int> bufC(sycl::range{count});

  composite_queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on({eventA, eventB});

    sycl::accessor acc(bufC, cgh, sycl::write_only);

    cgh.parallel_for(sycl::range{count},
                     [=](sycl::id<1> it) { acc[it] = ptrA[it] + ptrB[it]; });
  });

  component_queue.submit([&](sycl::handler& cgh) {
    sycl::accessor acc(bufC, cgh, sycl::read_write);

    cgh.parallel_for(sycl::range{count},
                     [=](sycl::id<1> it) { acc[it] += ptrA[it] + ptrB[it]; });
  });

  auto hostAcc = bufC.get_host_access();
  INFO("Verifying kernel (2 x vector add) results");
  for (size_t i = 0; i < count; ++i) {
    REQUIRE(2 * (i + i) == hostAcc[i]);
  }

#endif
}

}  // namespace composite_device::tests
