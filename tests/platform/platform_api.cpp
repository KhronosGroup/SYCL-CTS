/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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

#include "../common/common.h"

#define TEST_NAME platform_api

namespace platform_api__ {
using namespace sycl_cts;

// Compares two devices by their hash value.
struct DeviceHashLessT {
  bool operator()(const sycl::device& lDevice, const sycl::device& rDevice) {
    std::hash<sycl::device> hasher;
    return hasher(lDevice) < hasher(rDevice);
  }
};

// Checks that all devices in a vector are unique.
inline bool AllDevicesUnique(const std::vector<sycl::device>& devices) {
  std::vector<sycl::device> devicesCopy = devices;
  std::sort(devicesCopy.begin(), devicesCopy.end(), DeviceHashLessT{});
  return std::unique(devicesCopy.begin(), devicesCopy.end()) ==
         devicesCopy.end();
}

// Checks that all devices are in the list devices returned by the platform.
inline bool AllDevicesAreInPlatform(const std::vector<sycl::device>& devices,
                                    const sycl::platform& platform) {
  std::vector<sycl::device> devicesCopy = devices;
  std::vector<sycl::device> allDevices = platform.get_devices();
  std::sort(devicesCopy.begin(), devicesCopy.end(), DeviceHashLessT{});
  std::sort(allDevices.begin(), allDevices.end(), DeviceHashLessT{});
  return std::includes(allDevices.begin(), allDevices.end(),
                       devicesCopy.begin(), devicesCopy.end(),
                       DeviceHashLessT{});
}

// Checks that all devices return the specified device_type when queried.
inline bool AllDevicesHaveType(const std::vector<sycl::device>& devices,
                               sycl::info::device_type devType) {
  return std::all_of(
      devices.begin(), devices.end(), [devType](const sycl::device& device) {
        return device.get_info<sycl::info::device::device_type>() == devType;
      });
}

// Returns the number of devices in the platform with the specified device type.
inline size_t CountPlatformDevicesWithType(const sycl::platform& platform,
                                           sycl::info::device_type devType) {
  std::vector<sycl::device> allDevices = platform.get_devices();
  return std::count_if(
      allDevices.begin(), allDevices.end(),
      [devType](const sycl::device& device) {
        return device.get_info<sycl::info::device::device_type>() == devType;
      });
}

/** tests the api for sycl::platform
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    {
      /** check get_devices() member function
       */
      {
        INFO("Checking platform::get_devices()");
        auto plt = util::get_cts_object::platform(cts_selector);
        auto devs = plt.get_devices();
        check_return_type<std::vector<sycl::device>>(log, devs,
                                                     "platform::get_devices()");
        CHECK(AllDevicesUnique(devs));
      }

      /** check get_devices(info::device_type::all) member function
       */
      {
        INFO("Checking platform::get_devices(info::device_type::all)");
        auto plt = util::get_cts_object::platform(cts_selector);
        auto devs = plt.get_devices(sycl::info::device_type::all);
        check_return_type<std::vector<sycl::device>>(
            log, devs, "platform::get_devices(info::device_type::all)");
        CHECK(AllDevicesUnique(devs));
      }

      /** check get_devices(info::device_type::automatic) member function
       */
      {
        INFO("Checking platform::get_devices(info::device_type::automatic)");
        auto plt = util::get_cts_object::platform(cts_selector);
        auto devs = plt.get_devices(sycl::info::device_type::automatic);
        check_return_type<std::vector<sycl::device>>(
            log, devs, "platform::get_devices(info::device_type::automatic)");
        if (devs.size() == 0) {
          CHECK(plt.get_devices().size() == 0);
        } else {
          CHECK(AllDevicesAreInPlatform(devs, plt));
        }
      }

      /** check get_devices(info::device_type::<cpu|gpu|accelerator|custom>)
       * member function
       */
      for (sycl::info::device_type devType :
           {sycl::info::device_type::cpu, sycl::info::device_type::gpu,
            sycl::info::device_type::accelerator,
            sycl::info::device_type::custom}) {
        std::string devTypeName = [devType]() {
          switch (devType) {
            case sycl::info::device_type::cpu:
              return "sycl::info::device_type::cpu";
            case sycl::info::device_type::gpu:
              return "sycl::info::device_type::gpu";
            case sycl::info::device_type::accelerator:
              return "sycl::info::device_type::accelerator";
            case sycl::info::device_type::custom:
              return "sycl::info::device_type::custom";
            default:
              assert(false && "Missing enumeration!");
          }
        }();
        INFO("Checking platform::get_devices(" + devTypeName + ")");
        auto plt = util::get_cts_object::platform(cts_selector);
        auto devs = plt.get_devices(devType);
        check_return_type<std::vector<sycl::device>>(
            log, devs, "platform::get_devices("+ devTypeName + ")");
        CHECK(AllDevicesAreInPlatform(devs, plt));
        CHECK(AllDevicesHaveType(devs, devType));
        CHECK(devs.size() == CountPlatformDevicesWithType(plt, devType));
      }

      /** check has() member function
       */
      {
        auto plt = util::get_cts_object::platform(cts_selector);
        auto extensionSupported = plt.has(sycl::aspect::cpu);
        check_return_type<bool>(log, extensionSupported,
                                "platform::has(sycl::aspect)");
      }

      /** check has_extensions() member function
       */
      // TODO: mark this check as testing deprecated functionality
      {
        auto plt = util::get_cts_object::platform(cts_selector);
        auto extensionSupported = plt.has_extension(std::string("cl_khr_icd"));
        check_return_type<bool>(log, extensionSupported,
                                "platform::has_extension(string_class)");
      }

      /** check get_info() member function
       */
      {
        auto plt = util::get_cts_object::platform(cts_selector);
        auto platformName = plt.get_info<sycl::info::platform::name>();
        check_return_type<std::string>(log, platformName,
                                       "platform::get_info()");
      }

      /** check get_platforms() static method
       */
      {
        auto plt = sycl::platform::get_platforms();
        check_return_type<std::vector<sycl::platform>>(
            log, plt, "platform::get_platform()");
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace platform_api__ */
