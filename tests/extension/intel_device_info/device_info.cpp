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

#include "../../common/common.h"
#include <type_traits>

TEST_CASE("sycl::ext::intel::info::device::device_id", "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_device_id)) {
    auto ID = dev.get_info<sycl::ext::intel::info::device::device_id>();
    CHECK(std::is_same_v<decltype(ID), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_device_id.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::uuid", "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_device_info_uuid)) {
    auto UUID = dev.get_info<sycl::ext::intel::info::device::uuid>();
    CHECK(std::is_same_v<decltype(UUID), std::array<unsigned char, 16>>);
  } else {
    SKIP("Device does not support aspect::ext_intel_device_info_uuid.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::pci_address",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_pci_address)) {
    auto BDF = dev.get_info<sycl::ext::intel::info::device::pci_address>();
    CHECK(std::is_same_v<decltype(BDF), std::string>);
  } else {
    SKIP("Device does not support aspect::ext_intel_pci_address.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::gpu_eu_simd_width",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
    auto euSimdWidth =
        dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
    CHECK(std::is_same_v<decltype(euSimdWidth), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_gpu_eu_simd_width.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::gpu_eu_count",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_gpu_eu_count)) {
    auto euCount = dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
    CHECK(std::is_same_v<decltype(euCount), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_gpu_eu_count.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::gpu_slices", "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_gpu_slices)) {
    auto slices = dev.get_info<sycl::ext::intel::info::device::gpu_slices>();
    CHECK(std::is_same_v<decltype(slices), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_gpu_slices.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::gpu_subslices_per_slice",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
    auto subslices =
        dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
    CHECK(std::is_same_v<decltype(subslices), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_gpu_subslices_per_slice.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::gpu_eu_count_per_subslice",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
    auto euCount = dev.get_info<
        sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
    CHECK(std::is_same_v<decltype(euCount), uint32_t>);
  } else {
    SKIP(
        "Device does not support aspect::ext_intel_gpu_eu_count_per_subslice.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::gpu_hw_threads_per_eu",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
    auto threadsCount =
        dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    CHECK(std::is_same_v<decltype(threadsCount), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_gpu_hw_threads_per_eu.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::max_mem_bandwidth",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_max_mem_bandwidth)) {
    auto maxBW =
        dev.get_info<sycl::ext::intel::info::device::max_mem_bandwidth>();
    CHECK(std::is_same_v<decltype(maxBW), uint64_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_max_mem_bandwidth.");
  }
#endif
}

// NOTE: To prevent this case from being skipped, the test must be run with:
// ZES_ENABLE_SYSMAN=1 ./test_intel_device_info
// or
// ZES_ENABLE_SYSMAN=1 ./test_intel_device_info -#
// "sycl::ext::intel::info::device::free_memory"
TEST_CASE("sycl::ext::intel::info::device::free_memory",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_free_memory)) {
    auto GlobalMemSize = dev.get_info<sycl::info::device::global_mem_size>();
    auto FreeMemory =
        dev.get_info<sycl::ext::intel::info::device::free_memory>();
    CHECK(std::is_same_v<decltype(FreeMemory), uint64_t>);
    CHECK(FreeMemory > 0);
    CHECK(FreeMemory <= GlobalMemSize);

    // Сheck the free memory size after allocating 1Mb * sizeof(int) of memory
    // on the device.
    const size_t NumBytesToAlloc = 1024 * 1024 * sizeof(int);
    sycl::queue q(dev);
    int* p = static_cast<int*>(
        sycl::malloc_device(NumBytesToAlloc, dev, q.get_context()));
    auto FreeMemoryAfterAlloc =
        dev.get_info<sycl::ext::intel::info::device::free_memory>();
    CHECK(FreeMemoryAfterAlloc == FreeMemory - NumBytesToAlloc);

    // Сheck the free memory size after launching a kernel & filling this
    // memory.
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class fill_kernel>(
          sycl::range<1>(NumBytesToAlloc / sizeof(int)),
          [=](sycl::id<1> id) { p[id] = id; });
    });
    q.wait();
    auto FreeMemoryAfterKernel =
        dev.get_info<sycl::ext::intel::info::device::free_memory>();
    CHECK(FreeMemoryAfterKernel <= FreeMemoryAfterAlloc);

    // Сheck the free memory size after freeing memory.
    sycl::free(p, q.get_context());
    auto FreeMemoryAfterFree =
        dev.get_info<sycl::ext::intel::info::device::free_memory>();
    CHECK(FreeMemoryAfterFree <= FreeMemory);
  } else {
    SKIP("Device does not support aspect::ext_intel_free_memory.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::memory_clock_rate",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_memory_clock_rate)) {
    auto MemoryClockRate =
        dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
    CHECK(std::is_same_v<decltype(MemoryClockRate), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_memory_clock_rate.");
  }
#endif
}

TEST_CASE("sycl::ext::intel::info::device::memory_bus_width",
          "[intel_device_info]") {
#ifndef SYCL_EXT_INTEL_DEVICE_INFO
  SKIP("SYCL_EXT_INTEL_DEVICE_INFO is not defined");
#else
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_memory_bus_width)) {
    auto MemoryBusWidth =
        dev.get_info<sycl::ext::intel::info::device::memory_bus_width>();
    CHECK(std::is_same_v<decltype(MemoryBusWidth), uint32_t>);
  } else {
    SKIP("Device does not support aspect::ext_intel_memory_bus_width.");
  }
#endif
}
