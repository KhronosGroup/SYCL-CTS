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

#include "../common/common.h"
#include "../common/semantics_reference.h"

#include <vector>

struct storage {
  bool is_cpu;
  bool is_gpu;
  bool is_accelerator;
  bool has_fp16;
  bool has_fp64;

  explicit storage(const sycl::device& device)
      : is_cpu(device.is_cpu()),
        is_gpu(device.is_gpu()),
        is_accelerator(device.is_accelerator()),
        has_fp16(device.has(sycl::aspect::fp16)),
        has_fp64(device.has(sycl::aspect::fp64)) {}

  bool check(const sycl::device& device) const {
    return device.is_cpu() == is_cpu && device.is_gpu() == is_gpu &&
           device.is_accelerator() == is_accelerator &&
           device.has(sycl::aspect::fp16) == has_fp16 &&
           device.has(sycl::aspect::fp64) == has_fp64;
  }
};

TEST_CASE("device common reference semantics", "[device]") {
  sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::platform platform = sycl_cts::util::get_cts_object::platform();
  const std::vector<sycl::device> devices = platform.get_devices();

  // obtain and test with a distinct device, if possible
  if (devices.size() > 1) {
    for (const auto& other_device : devices) {
      if (device != other_device) {
        INFO("using two distinct devices");
        common_reference_semantics::check_host<storage>(device, other_device,
                                                        "device");
        return;  // test is finished, single device test is a subset of this
      }
    }
  }

  // else, test with a single device
  INFO("using a single device");
  common_reference_semantics::check_host<storage>(device, "device");
}
