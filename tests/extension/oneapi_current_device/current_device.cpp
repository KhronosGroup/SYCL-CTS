/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
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
#include <thread>

namespace current_device::tests {

TEST_CASE(
    "Test for \"get_current_device\" get default device, call function and "
    "compare devices ") {
#ifndef SYCL_EXT_ONEAPI_CURRENT_DEVICE
  SKIP(
      "The sycl_ext_oneapi_current_device extension is not supported "
      "by an implementation");
#else
  if (sycl::device::get_devices().size() < 1) {
    SKIP("Test requires at least one device");
  }
  const sycl::device default_device{sycl::default_selector_v};
  const auto current_device =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  CHECK(
      std::is_same_v<std::remove_cv_t<decltype(current_device)>, sycl::device>);
  CHECK(default_device == current_device);
#endif
}

TEST_CASE(
    "Test for \"set_current_device and get_current_device\" checks that "
    "get_current_device function returns the device set by "
    "set_current_device function") {
#ifndef SYCL_EXT_ONEAPI_CURRENT_DEVICE
  SKIP(
      "The sycl_ext_oneapi_current_device extension is not supported by an "
      "implementation");
#else
  for (const auto& device : sycl::device::get_devices()) {
    sycl::ext::oneapi::experimental::this_thread::set_current_device(device);
    const auto current_device =
        sycl::ext::oneapi::experimental::this_thread::get_current_device();
    CHECK(std::is_same_v<std::remove_cv_t<decltype(current_device)>,
                         sycl::device>);
    CHECK(device == current_device);
  }
#endif
}

void thread_callback(const sycl::device& device_to_set,
                     sycl::device& current_device) {
  sycl::ext::oneapi::experimental::this_thread::set_current_device(
      device_to_set);
  current_device =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
}

TEST_CASE(
    "Test for calling \"set_current_device and get_current_device in "
    "different threads\", checks that each thread has its own current device") {
#ifndef SYCL_EXT_ONEAPI_CURRENT_DEVICE
  SKIP(
      "The sycl_ext_oneapi_current_device extension is not supported "
      "by an implementation");
#else
  const auto devices = sycl::device::get_devices();
  if (devices.size() < 2) {
    SKIP("Test requires at least two devices");
  }
  const auto t1_device_to_set = devices[0];
  const auto t2_device_to_set = devices[1];
  sycl::device t1_current_device;
  sycl::device t2_current_device;

  std::thread t1(thread_callback, std::cref(t1_device_to_set),
                 std::ref(t1_current_device));
  std::thread t2(thread_callback, std::cref(t2_device_to_set),
                 std::ref(t2_current_device));
  t1.join();
  t2.join();

  CHECK(t1_current_device == t1_device_to_set);
  CHECK(t2_current_device == t2_device_to_set);
  CHECK(t1_current_device != t2_current_device);

#endif
}
}  // namespace current_device::tests
