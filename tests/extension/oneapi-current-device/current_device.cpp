/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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

#include "../../common/common.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace current_device {
#ifdef SYCL_EXT_ONEAPI_CURRENT_DEVICE

sycl::device get_device_to_set() {
  static std::vector<sycl::device> devices = sycl::device::get_devices();
  for (const auto device : devices) {
    // remove device from the list of devices
    devices.erase(std::remove(devices.begin(), devices.end(), device),
                  devices.end());
    return device;
  }
  SKIP("Test did not find any device to set as current");
}

#endif

TEST_CASE(
    "Test for \"get_current_device\" extension, get default device, call "
    "function and compare devices ") {
#ifndef SYCL_EXT_ONEAPI_CURRENT_DEVICE
  SKIP("SYCL_EXT_ONEAPI_CURRENT_DEVICE is not defined");
#else
  if (sycl::device::get_devices().size() < 1) {
    SKIP("Test requires at least one device");
  }
  const sycl::device default_device{sycl::default_selector()};
  const auto current_device =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  CHECK(
      std::is_same_v<std::remove_cv_t<decltype(current_device)>, sycl::device>);
  CHECK(default_device == current_device);
#endif
}

TEST_CASE(
    "Test for \"set_current_device and get_current_device\" extension, check "
    "that get function returns the device set by set function") {
#ifndef SYCL_EXT_ONEAPI_CURRENT_DEVICE
  SKIP("SYCL_EXT_ONEAPI_CURRENT_DEVICE is not defined");
#else
  const auto device_to_set = current_device::get_device_to_set();
  sycl::ext::oneapi::experimental::this_thread::set_current_device(
      device_to_set);
  const auto current_device =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  CHECK(
      std::is_same_v<std::remove_cv_t<decltype(current_device)>, sycl::device>);
  CHECK(device_to_set == current_device);
#endif
}

TEST_CASE(
    "Test for \"set_current_device and get_current_device in different "
    "threads\" extension, check "
    "that each thread has its own current device") {
#ifndef SYCL_EXT_ONEAPI_CURRENT_DEVICE
  SKIP("SYCL_EXT_ONEAPI_CURRENT_DEVICE is not defined");
#else
  if (sycl::device::get_devices().size() < 2) {
    SKIP("Test requires at least two devices");
  }
  std::mutex mtx;
  std::condition_variable cv_thread1;
  std::condition_variable cv_thread2;
  bool signal_thread1 = false;
  bool signal_thread2 = false;

  const auto t1_device_to_set = get_device_to_set();
  const auto t2_device_to_set = get_device_to_set();
  sycl::device t1_current_device;
  sycl::device t2_current_device;

  std::thread t1([&]() {
    {
      std::unique_lock<std::mutex> lock(mtx);
      sycl::ext::oneapi::experimental::this_thread::set_current_device(
          t1_device_to_set);
      signal_thread1 = true;
    }
    cv_thread1.notify_one();
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv_thread2.wait(lock, [&] { return signal_thread2; });
    }
    const auto t1_current_device =
        sycl::ext::oneapi::experimental::this_thread::get_current_device();
  });

  std::thread t2([&]() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_thread1.wait(lock, [&] { return signal_thread1; });
    sycl::ext::oneapi::experimental::this_thread::set_current_device(
        t2_device_to_set);
    signal_thread2 = true;
    const auto t2_current_device =
        sycl::ext::oneapi::experimental::this_thread::get_current_device();
  });
  t1.join();
  t2.join();

  CHECK(t1_current_device == t1_device_to_set);
  CHECK(t2_current_device == t2_device_to_set);
  CHECK(t1_current_device != t2_current_device);

#endif
}

}  // namespace current_device