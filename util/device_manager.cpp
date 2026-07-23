/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2021-2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "device_manager.h"

#include <fstream>

#include "../tests/common/cts_selector.h"

namespace sycl_cts {
namespace util {

static std::string get_device_type_str(const sycl::device& d) {
  const auto deviceType = d.get_info<sycl::info::device::device_type>();
  switch (deviceType) {
    case sycl::info::device_type::host:
      return "host";
    case sycl::info::device_type::cpu:
      return "cpu";
    case sycl::info::device_type::gpu:
      return "gpu";
    case sycl::info::device_type::accelerator:
      return "accelerator";
    case sycl::info::device_type::custom:
      return "custom";
    case sycl::info::device_type::automatic:
      return "automatic";
    case sycl::info::device_type::all:
      return "all";
    default:
      assert(false);
      return "(unknown)";
  };
}

void device_manager::list_devices() const {
  const auto all_devices = sycl::device::get_devices();
  const auto cts_device = sycl::device(cts_selector);

  if (all_devices.empty()) {
    printf("No devices available.\n");
    return;
  }

  printf("%zu devices available (> = currently selected):\n",
         all_devices.size());
  printf("  %-12s %s\n", "Type", "Platform / Device");
  for (auto& d : all_devices) {
    const auto deviceType = get_device_type_str(d);
    const auto deviceName = d.get_info<sycl::info::device::name>();
    const auto platformName =
        d.get_platform().get_info<sycl::info::platform::name>();
    printf("%c ", d == cts_device ? '>' : ' ');
    printf("%-12s %s / %s\n", deviceType.c_str(), platformName.c_str(),
           deviceName.c_str());
  }
}

void device_manager::dump_info(const std::string& infoDumpFile) {
  auto chosenDevice = sycl::device(cts_selector);
  auto chosenPlatform = sycl::platform(cts_selector);

  std::fstream infoFile(infoDumpFile, std::ios::out);

  auto deviceNameStr = chosenDevice.get_info<sycl::info::device::name>();
  auto deviceVendorStr = chosenDevice.get_info<sycl::info::device::vendor>();
  auto deviceType = chosenDevice.get_info<sycl::info::device::device_type>();
  auto deviceVersionStr = chosenDevice.get_info<sycl::info::device::version>();
  std::string deviceTypeStr;
  switch (deviceType) {
    case sycl::info::device_type::host:
      deviceTypeStr = "device_type::host";
      break;
    case sycl::info::device_type::cpu:
      deviceTypeStr = "device_type::cpu";
      break;
    case sycl::info::device_type::gpu:
      deviceTypeStr = "device_type::gpu";
      break;
    case sycl::info::device_type::accelerator:
      deviceTypeStr = "device_type::accelerator";
      break;
    case sycl::info::device_type::custom:
      deviceTypeStr = "device_type::custom";
      break;
    case sycl::info::device_type::automatic:
      deviceTypeStr = "device_type::automatic";
      break;
    case sycl::info::device_type::all:
      deviceTypeStr = "device_type::all";
      break;
  };
  auto doesDeviceSupportHalf =
      chosenDevice.has(sycl::aspect::fp16) ? "Supported" : "Not Supported";
  auto doesDeviceSupportDouble =
      chosenDevice.has(sycl::aspect::fp64) ? "Supported" : "Not Supported";
  auto doesDeviceSupportAtomics =
      chosenDevice.has(sycl::aspect::atomic64) ? "Supported" : "Not Supported";
  auto platformNameStr = chosenPlatform.get_info<sycl::info::platform::name>();
  auto platformVendorStr =
      chosenPlatform.get_info<sycl::info::platform::vendor>();
  auto platformVersionStr =
      chosenPlatform.get_info<sycl::info::platform::version>();

  infoFile << "{\"device-name\": \"" << deviceNameStr
           << "\", \"device-vendor\": \"" << deviceVendorStr
           << "\", \"device-type\": \"" << deviceTypeStr
           << "\", \"device-version\": \"" << deviceVersionStr
           << "\", \"device-fp16\": \"" << doesDeviceSupportHalf
           << "\", \"device-fp64\": \"" << doesDeviceSupportDouble
           << "\", \"device-atomic64\": \"" << doesDeviceSupportAtomics
           << "\", \"platform-name\": \"" << platformNameStr
           << "\", \"platform-vendor\": \"" << platformVendorStr
           << "\", \"platform-version\": \"" << platformVersionStr << "\"}";
}

}  // namespace util
}  // namespace sycl_cts
