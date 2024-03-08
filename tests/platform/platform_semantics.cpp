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
#include "../common/disabled_for_test_case.h"
#ifndef SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "../common/semantics_reference.h"
#endif

#include <vector>

struct storage {
  size_t device_count;
  bool is_cpu;
  bool is_gpu;
  bool is_accelerator;
  bool has_fp16;
  bool has_fp64;

  explicit storage(const sycl::platform& platform)
      : device_count(platform.get_devices().size()),
        is_cpu(platform.has(sycl::aspect::cpu)),
        is_gpu(platform.has(sycl::aspect::gpu)),
        is_accelerator(platform.has(sycl::aspect::accelerator)),
        has_fp16(platform.has(sycl::aspect::fp16)),
        has_fp64(platform.has(sycl::aspect::fp64)) {}

  bool check(const sycl::platform& platform) const {
    return platform.get_devices().size() == device_count &&
           platform.has(sycl::aspect::cpu) == is_cpu &&
           platform.has(sycl::aspect::gpu) == is_gpu &&
           platform.has(sycl::aspect::accelerator) == is_accelerator &&
           platform.has(sycl::aspect::fp16) == has_fp16 &&
           platform.has(sycl::aspect::fp64) == has_fp64;
  }
};

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("platform common reference semantics", "[platform]")({
  sycl::platform platform = sycl_cts::util::get_cts_object::platform();
  const std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

  // obtain and test with a distinct platform, if possible
  if (platforms.size() > 1) {
    for (const auto& other_platform : platforms) {
      if (platform != other_platform) {
        INFO("using two distinct platforms");
        common_reference_semantics::check_host<storage>(
            platform, other_platform, "platform");
        return;  // test is finished, single platform test is a subset of this
      }
    }
  }

  // else, test with a single platform
  INFO("using a single platform");
  common_reference_semantics::check_host<storage>(platform, "platform");
});
