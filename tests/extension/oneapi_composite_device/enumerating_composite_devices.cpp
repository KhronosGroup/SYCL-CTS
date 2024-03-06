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

TEST_CASE("Test for signatures of the new get_composite_device APIs",
          "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "implementation");
#else

  STATIC_REQUIRE(
      std::is_same_v<
          std::vector<sycl::device>,
          decltype(sycl::ext::oneapi::experimental::get_composite_devices())>);

  auto platform = sycl_cts::util::get_cts_object::platform();

  STATIC_REQUIRE(
      std::is_same_v<std::vector<sycl::device>,
                     decltype(platform.ext_oneapi_get_composite_devices())>);

#endif
}

TEST_CASE("Test for free function get_composite_device",
          "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "implementation");
#else

  std::vector<sycl::device> composite_devices;

  // get_composite_devices may not throw
  try {
    composite_devices =
        sycl::ext::oneapi::experimental::get_composite_devices();
  } catch (sycl::exception& e) {
    FAIL("get_composite_devices threw an exception: " << e.what());
  }

  {
    INFO("Subsequent calls to get_composite_devices()");
    std::vector<sycl::device> composite_devices2;
    // get_composite_devices may not throw
    try {
      composite_devices2 =
          sycl::ext::oneapi::experimental::get_composite_devices();
      INFO("Checking that subsequent call returns the same list of devices");
      REQUIRE(composite_devices == composite_devices2);
    } catch (sycl::exception& e) {
      FAIL("get_composite_devices threw an exception: " << e.what());
    }
  }

#endif
}

TEST_CASE("Test for platform::ext_oneapi_get_composite_devices",
          "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "implementation");
#else

  auto platform = sycl_cts::util::get_cts_object::platform();
  std::vector<sycl::device> composite_devices;

  // ext_oneapi_get_composite_devices may not throw
  try {
    composite_devices = platform.ext_oneapi_get_composite_devices();
  } catch (sycl::exception& e) {
    FAIL("platform::ext_oneapi_get_composite_devices threw an exception: "
         << e.what());
  }

  {
    INFO("Subsequent calls to platform::ext_oneapi_get_composite_devices()");
    std::vector<sycl::device> composite_devices2;
    // ext_oneapi_get_composite_devices may not throw
    try {
      composite_devices2 = platform.ext_oneapi_get_composite_devices();
      INFO("Checking that subsequent call returns the same list of devices");
      REQUIRE(composite_devices == composite_devices2);
    } catch (sycl::exception& e) {
      FAIL("platform::ext_oneapi_get_composite_devices threw an exception: "
           << e.what());
    }
  }

  INFO(
      "Checking that devices returned by "
      "platform::ext_oneapi_get_composite_devices are also returned by "
      "get_composite_devices");
  auto all_composite_devices =
      sycl::ext::oneapi::experimental::get_composite_devices();
  for (auto device_p : composite_devices) {
    // device_p is a composite device returned by
    // platform::ext_oneapi_get_composite_devices
    REQUIRE(std::any_of(
        all_composite_devices.begin(), all_composite_devices.end(),
        [&](sycl::device device_d) { return device_d == device_p; }));
  }

#endif
}

TEST_CASE(
    "Test to check that composite devices are not returned as root devices",
    "[oneapi_composite_device]") {
#ifndef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
  SKIP(
      "The sycl_ext_oneapi_composite device extension is not supported by an "
      "implementation");
#else

  std::vector<sycl::device> root_devices = sycl::device::get_devices();
  // We do not check platform::get_devices(), because it is expected to return
  // a subset of device::get_devices()

  std::vector<sycl::device> composite_devices =
      sycl::ext::oneapi::experimental::get_composite_devices();
  // We do not check platform::ext_oneapi_get_composite_devices(), because it is
  // expected to return a subset of get_composite_devices()

  for (auto composite_device : composite_devices) {
    REQUIRE(std::none_of(root_devices.begin(), root_devices.end(),
                         [&](sycl::device& root_device) {
                           return composite_device != root_device;
                         }));
  }
#endif
}

}  // namespace composite_device::tests
