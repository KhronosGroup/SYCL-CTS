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
#include <type_traits>
#include <vector>

namespace composite_device::tests {

TEST_CASE("Test for aspects of composite devices",
          "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "imlementation");
#else
  auto platform = sycl_cts::util::get_cts_object::platform();
  auto platform_composite_devices = platform.ext_oneapi_get_composite_devices();
  auto all_composite_devices =
      sycl::ext::oneapi::experimental::get_composite_devices();

  INFO(
      "Checking that all devices returned by platform::get_composite_devices() "
      "and by sycl::ext::oneapi::experimental::get_composite_devices() have "
      "sycl_ext_oneapi_is_composite aspect");
  REQUIRE(std::all_of(
      platform_composite_devices.begin(), platform_composite_devices.end(),
      [&](sycl::device composite_device) {
        return composite_device.has(sycl::aspect::ext_oneapi_is_composite);
      }));
  REQUIRE(std::all_of(
      all_composite_devices.begin(), all_composite_devices.end(),
      [&](sycl::device composite_device) {
        return composite_device.has(sycl::aspect::ext_oneapi_is_composite);
      }));

  INFO(
      "Checking that none of devices returned by "
      "platform::get_composite_devices() "
      "and by sycl::ext::oneapi::experimental::get_composite_devices() have "
      "sycl_ext_oneapi_is_component aspect");
  REQUIRE(std::none_of(
      platform_composite_devices.begin(), platform_composite_devices.end(),
      [&](sycl::device composite_device) {
        return composite_device.has(sycl::aspect::ext_oneapi_is_component);
      }));
  REQUIRE(std::none_of(
      all_composite_devices.begin(), all_composite_devices.end(),
      [&](sycl::device composite_device) {
        return composite_device.has(sycl::aspect::ext_oneapi_is_component);
      }));

#endif
}

TEST_CASE("Test for ext_oneapi_is_component aspect",
          "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "imlementation");
#else

  auto device = sycl_cts::util::get_cts_object::device();

  if (!device.has(sycl::aspect::ext_oneapi_is_component))
    SKIP(
        "The selected device doesn't have ext_oneapi_is_component aspect, this "
        "test has nothing to check");

  INFO(
      "Checking that a component device is not considered a composite device "
      "at the same time");
  REQUIRE(!device.has(sycl::aspect::ext_oneapi_is_composite));

  std::vector<sycl::info::partition_property> supported_partitions =
      device.get_info<sycl::info::device::partition_properties>();
  INFO(
      "Checking that sub-devices of a component device are not considered to "
      "be component devices");
  for (auto selected_partition : supported_partitions) {
    switch (selected_partition) {
      case sycl::info::partition_property::partition_equally: {
        constexpr size_t count = 2;  // Guaranteed to work by SYCL 2020 spec
        auto sub_devices = device.create_sub_devices<
            sycl::info::partition_property::partition_equally>(count);
        REQUIRE(std::none_of(sub_devices.begin(), sub_devices.end(),
                             [&](sycl::device sub_device) {
                               return sub_device.has(
                                   sycl::aspect::ext_oneapi_is_component);
                             }));

      } break;
      case sycl::info::partition_property::partition_by_counts: {
        const std::vector<size_t> counts = {1, 1};
        auto sub_devices = device.create_sub_devices<
            sycl::info::partition_property::partition_by_counts>(counts);
        REQUIRE(std::none_of(sub_devices.begin(), sub_devices.end(),
                             [&](sycl::device sub_device) {
                               return sub_device.has(
                                   sycl::aspect::ext_oneapi_is_component);
                             }));
      } break;
      case sycl::info::partition_property::partition_by_affinity_domain: {
        auto supported_domains =
            device.get_info<sycl::info::device::partition_affinity_domains>();
        for (auto domain : supported_domains) {
          auto sub_devices = device.create_sub_devices<
              sycl::info::partition_property::partition_by_affinity_domain>(
              domain);
          REQUIRE(
              std::none_of(sub_devices.begin(), sub_devices.end(),
                           [&](sycl::device sub_device) {
                             return sub_device.has(
                                 sycl::aspect::ext_oneapi_is_component);
                           }));
        }
      } break;
      default:
        // Unknown partition type, do nothing
        ;
    }
  }

#endif
}

}  // namespace composite_device::tests

