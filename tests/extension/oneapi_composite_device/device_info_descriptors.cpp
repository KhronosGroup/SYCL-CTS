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

#include <type_traits>
#include <vector>

namespace composite_device::tests {

TEST_CASE("Test for info::device::component_devices descriptor",
          "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "imlementation");
#else

  std::vector<sycl::device> devices_to_test;
  devices_to_test.push_back(sycl_cts::util::get_cts_object::device());

  auto platform = devices_to_test.front().get_platform();

  for (auto device : platform.ext_oneapi_get_composite_devices()) {
    devices_to_test.push_back(device);
  }

  for (auto device : devices_to_test) {
    STATIC_REQUIRE(
        std::is_same_v<
            std::vector<sycl::device>,
            decltype(device.get_info<sycl::ext::oneapi::experimental::info::
                                         device::component_devices>())>);

    auto result = device.get_info<
        sycl::ext::oneapi::experimental::info::device::component_devices>();

    if (!device.has(sycl::aspect::ext_oneapi_is_composite)) {
      INFO(
          "Checking that non-composite device doesn't have any component "
          "devices");
      REQUIRE(result.empty());
    } else {
      INFO(
          "Composite devices are expected to have at least two component "
          "devices");
      REQUIRE(result.size() >= 2);

      for (auto component_device : result) {
        INFO("Each component device should have corresponding aspect");
        REQUIRE(component_device.has(sycl::aspect::ext_oneapi_is_component));
        INFO(
            "Each component device should return the correct composite device");
        auto composite_device = component_device.get_info<
            sycl::ext::oneapi::experimental::info::device::composite_device>();
        REQUIRE(composite_device == device);
      }
    }
  }

#endif
}

TEST_CASE("Test for info::device::composite_device descriptor",
          "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "imlementation");
#else

  std::vector<sycl::device> devices_to_test;
  devices_to_test.push_back(sycl_cts::util::get_cts_object::device());

  auto platform = devices_to_test.front().get_platform();

  for (auto device : platform.ext_oneapi_get_composite_devices()) {
    devices_to_test.push_back(device);
  }

  for (auto device : devices_to_test) {
    STATIC_REQUIRE(
        std::is_same_v<
            sycl::device,
            decltype(device.get_info<sycl::ext::oneapi::experimental::info::
                                         device::composite_device>())>);

    try {
      auto result = device.get_info<
          sycl::ext::oneapi::experimental::info::device::composite_device>();
      if (!device.has(sycl::aspect::ext_oneapi_is_component)) {
        FAIL(
            "Querying a composite device should result in exception if query "
            "is performed for a non-component device");
      }

      REQUIRE(result.has(sycl::aspect::ext_oneapi_is_composite));
    } catch (const sycl::exception& e) {
      if (!device.has(sycl::aspect::ext_oneapi_is_component)) {
        INFO(
            "Checking that a non-component device don't have a composite "
            "device associcated with it");
        REQUIRE(e.code() == sycl::errc::invalid);
      } else {
        FAIL(
            "Unexpected exception when querying for a composite device that "
            "a component device belongs to: "
            << e.what());
      }
    }
  }

#endif
}

}  // namespace composite_device::tests
