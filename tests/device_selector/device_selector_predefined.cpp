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

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"

namespace device_selector_predefined {
using namespace sycl_cts;

bool is_host(const sycl::device &device) {
  return device.get_info<sycl::info::device::device_type>() !=
         sycl::info::device_type::host;
}

TEST_CASE("predefined selectors", "[device_selector]") {
  bool gpuAvailable = false;
  bool cpuAvailable = false;
  bool acceleratorAvailable = false;
  auto devices = sycl::device::get_devices();
  for (const auto &device : devices) {
    if (device.is_gpu()) {
      gpuAvailable = true;
    }
    if (device.is_cpu()) {
      cpuAvailable = true;
    }
    if (device.is_accelerator()) {
      acceleratorAvailable = true;
    }
  }
  const bool noneAvailable =
      !cpuAvailable && !gpuAvailable && !acceleratorAvailable;

  // Compatibility with old SYCL 1.2.1 device selectors.

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  {  // default_selector
    if (noneAvailable) {
      sycl::default_selector defaultSelector;
      try {
        auto defaultDevice = util::get_cts_object::device(defaultSelector);
        if (is_host(defaultDevice)) {
          FAIL("selected a non-host device when no devices are available");
        }
      } catch (const sycl::exception &e) {
        FAIL("failed to select a host device when no devices are available");
      }
    }
  }
  {  // cpu_selector
    sycl::cpu_selector cpuSelector;
    if (cpuAvailable) {
      auto cpuDevice = util::get_cts_object::device(cpuSelector);
      CHECK(cpuDevice.is_cpu());
    } else {
      try {
        auto cpuDevice = util::get_cts_object::device(cpuSelector);
        FAIL("selected a CPU device when none are available");
      } catch (const sycl::exception &e) {
        // sycl::exception is expected
      } catch (...) {
        FAIL("wrong error thrown when no CPU devices are available");
      }
    }
  }
  {  // gpu_selector
    sycl::gpu_selector gpuSelector;
    if (gpuAvailable) {
      auto gpuDevice = util::get_cts_object::device(gpuSelector);
      CHECK(gpuDevice.is_gpu());
    } else {
      try {
        auto gpuDevice = util::get_cts_object::device(gpuSelector);
        FAIL("selected a GPU device when none are available");
      } catch (const sycl::exception &e) {
        // sycl::exception is expected
      } catch (...) {
        FAIL("wrong error thrown when no GPU devices are available");
      }
    }
  }
  {  // accelerator_selector
    sycl::accelerator_selector acceleratorSelector;
    if (acceleratorAvailable) {
      auto acceleratorDevice =
          util::get_cts_object::device(acceleratorSelector);
      CHECK(acceleratorDevice.is_accelerator());
    } else {
      try {
        auto acceleratorDevice =
            util::get_cts_object::device(acceleratorSelector);
        FAIL("selected an accelerator device when none are available");
      } catch (const sycl::exception &e) {
        // sycl::exception is expected
      } catch (...) {
        FAIL("wrong error thrown when no accelerator devices are available");
      }
    }
  }
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

  // SYCL2020 device selectors.

  {  // default_selector_v
    if (noneAvailable) {
      try {
        auto defaultDevice =
            util::get_cts_object::device(sycl::default_selector_v);
        if (is_host(defaultDevice)) {
          FAIL("selected non-host device when no devices are available");
        }
      } catch (const sycl::exception& e) {
        CHECK((sycl::errc::runtime == e.code()));
      }
    }
  }
  {  // cpu_selector_v
    const auto cpuSelector = sycl::cpu_selector_v;
    if (cpuAvailable) {
      auto cpuDevice = util::get_cts_object::device(cpuSelector);
      CHECK(cpuDevice.is_cpu());
    } else {
      try {
        auto cpuDevice = util::get_cts_object::device(cpuSelector);
        FAIL("selected a CPU device when none are available");
      } catch (const sycl::exception& e) {
        CHECK((sycl::errc::runtime == e.code()));
      }
    }
  }
  {  // gpu_selector_v
    const auto gpuSelector = sycl::gpu_selector_v;
    if (gpuAvailable) {
      auto gpuDevice = util::get_cts_object::device(gpuSelector);
      CHECK(gpuDevice.is_gpu());
    } else {
      try {
        auto gpuDevice = util::get_cts_object::device(gpuSelector);
        FAIL("selected a GPU device when none are available");
      } catch (const sycl::exception& e) {
        CHECK((sycl::errc::runtime == e.code()));
      }
    }
  }
  {  // accelerator_selector_v
    const auto acceleratorSelector = sycl::accelerator_selector_v;
    if (acceleratorAvailable) {
      auto acceleratorDevice =
          util::get_cts_object::device(acceleratorSelector);
      CHECK(acceleratorDevice.is_accelerator());
    } else {
      try {
        auto acceleratorDevice =
            util::get_cts_object::device(acceleratorSelector);
        FAIL("selected an accelerator device when none are available");
      } catch (const sycl::exception& e) {
        CHECK(sycl::errc::runtime == e.code());
      }
    }
  }
}

}  // namespace device_selector_predefined
