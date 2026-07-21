/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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
