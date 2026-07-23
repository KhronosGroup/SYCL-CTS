/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2026 The Khronos Group Inc.
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

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/device.hpp>
#include <type_traits>

namespace khr_split_headers::tests {

TEST_CASE("the device header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][device]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the device header defines the default_selector_v variable",
          "[khr_split_headers][device]") {
  using selector_t = decltype(sycl::default_selector_v);
  STATIC_REQUIRE(!std::is_void_v<selector_t>);
}

TEST_CASE("the device header defines the gpu_selector_v variable",
          "[khr_split_headers][device]") {
  using selector_t = decltype(sycl::gpu_selector_v);
  STATIC_REQUIRE(!std::is_void_v<selector_t>);
}

TEST_CASE("the device header defines the accelerator_selector_v variable",
          "[khr_split_headers][device]") {
  using selector_t = decltype(sycl::accelerator_selector_v);
  STATIC_REQUIRE(!std::is_void_v<selector_t>);
}

TEST_CASE("the device header defines the cpu_selector_v variable",
          "[khr_split_headers][device]") {
  using selector_t = decltype(sycl::cpu_selector_v);
  STATIC_REQUIRE(!std::is_void_v<selector_t>);
}

TEST_CASE("the device header defines the aspect_selector variable",
          "[khr_split_headers][device]") {
  using selector_t = decltype(sycl::aspect_selector());
  STATIC_REQUIRE(!std::is_void_v<selector_t>);
}

#define SYCL_DEVICE_DESCRIPTOR_TEST(descriptor)                                \
  TEST_CASE("the device header defines the " #descriptor " device descriptor", \
            "[khr_split_headers][device]") {                                   \
    STATIC_REQUIRE(                                                            \
        sycl_cts::util::is_complete_v<sycl::info::device::descriptor>);        \
  }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
#define SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(descriptor) \
  SYCL_DEVICE_DESCRIPTOR_TEST(descriptor)
#else
#define SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(descriptor)
#endif

SYCL_DEVICE_DESCRIPTOR_TEST(device_type)
SYCL_DEVICE_DESCRIPTOR_TEST(vendor_id)
SYCL_DEVICE_DESCRIPTOR_TEST(max_compute_units)
SYCL_DEVICE_DESCRIPTOR_TEST(max_work_item_dimensions)
SYCL_DEVICE_DESCRIPTOR_TEST(max_work_item_sizes<1>)
SYCL_DEVICE_DESCRIPTOR_TEST(max_work_item_sizes<2>)
SYCL_DEVICE_DESCRIPTOR_TEST(max_work_item_sizes<3>)
SYCL_DEVICE_DESCRIPTOR_TEST(max_work_group_size)
SYCL_DEVICE_DESCRIPTOR_TEST(max_num_sub_groups)
SYCL_DEVICE_DESCRIPTOR_TEST(sub_group_sizes)
SYCL_DEVICE_DESCRIPTOR_TEST(preferred_vector_width_char)
SYCL_DEVICE_DESCRIPTOR_TEST(preferred_vector_width_short)
SYCL_DEVICE_DESCRIPTOR_TEST(preferred_vector_width_int)
SYCL_DEVICE_DESCRIPTOR_TEST(preferred_vector_width_long)
SYCL_DEVICE_DESCRIPTOR_TEST(preferred_vector_width_float)
SYCL_DEVICE_DESCRIPTOR_TEST(preferred_vector_width_double)
SYCL_DEVICE_DESCRIPTOR_TEST(preferred_vector_width_half)
SYCL_DEVICE_DESCRIPTOR_TEST(native_vector_width_char)
SYCL_DEVICE_DESCRIPTOR_TEST(native_vector_width_short)
SYCL_DEVICE_DESCRIPTOR_TEST(native_vector_width_int)
SYCL_DEVICE_DESCRIPTOR_TEST(native_vector_width_long)
SYCL_DEVICE_DESCRIPTOR_TEST(native_vector_width_float)
SYCL_DEVICE_DESCRIPTOR_TEST(native_vector_width_double)
SYCL_DEVICE_DESCRIPTOR_TEST(native_vector_width_half)
SYCL_DEVICE_DESCRIPTOR_TEST(max_clock_frequency)
SYCL_DEVICE_DESCRIPTOR_TEST(address_bits)
SYCL_DEVICE_DESCRIPTOR_TEST(max_mem_alloc_size)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(image_support)
SYCL_DEVICE_DESCRIPTOR_TEST(max_read_image_args)
SYCL_DEVICE_DESCRIPTOR_TEST(max_write_image_args)
SYCL_DEVICE_DESCRIPTOR_TEST(image2d_max_width)
SYCL_DEVICE_DESCRIPTOR_TEST(image2d_max_height)
SYCL_DEVICE_DESCRIPTOR_TEST(image3d_max_width)
SYCL_DEVICE_DESCRIPTOR_TEST(image3d_max_height)
SYCL_DEVICE_DESCRIPTOR_TEST(image3d_max_depth)
SYCL_DEVICE_DESCRIPTOR_TEST(image_max_buffer_size)
SYCL_DEVICE_DESCRIPTOR_TEST(max_samplers)
SYCL_DEVICE_DESCRIPTOR_TEST(max_parameter_size)
SYCL_DEVICE_DESCRIPTOR_TEST(mem_base_addr_align)
SYCL_DEVICE_DESCRIPTOR_TEST(half_fp_config)
SYCL_DEVICE_DESCRIPTOR_TEST(single_fp_config)
SYCL_DEVICE_DESCRIPTOR_TEST(double_fp_config)
SYCL_DEVICE_DESCRIPTOR_TEST(global_mem_cache_type)
SYCL_DEVICE_DESCRIPTOR_TEST(global_mem_cache_line_size)
SYCL_DEVICE_DESCRIPTOR_TEST(global_mem_cache_size)
SYCL_DEVICE_DESCRIPTOR_TEST(global_mem_size)
SYCL_DEVICE_DESCRIPTOR_TEST(max_constant_buffer_size)
SYCL_DEVICE_DESCRIPTOR_TEST(max_constant_args)
SYCL_DEVICE_DESCRIPTOR_TEST(local_mem_type)
SYCL_DEVICE_DESCRIPTOR_TEST(local_mem_size)
SYCL_DEVICE_DESCRIPTOR_TEST(error_correction_support)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(host_unified_memory)
SYCL_DEVICE_DESCRIPTOR_TEST(atomic_memory_order_capabilities)
SYCL_DEVICE_DESCRIPTOR_TEST(atomic_fence_order_capabilities)
SYCL_DEVICE_DESCRIPTOR_TEST(atomic_memory_scope_capabilities)
SYCL_DEVICE_DESCRIPTOR_TEST(atomic_fence_scope_capabilities)
SYCL_DEVICE_DESCRIPTOR_TEST(profiling_timer_resolution)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(is_endian_little)
SYCL_DEVICE_DESCRIPTOR_TEST(is_available)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(is_compiler_available)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(is_linker_available)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(execution_capabilities)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(queue_profiling)
SYCL_DEVICE_DESCRIPTOR_TEST(built_in_kernel_ids)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(built_in_kernels)
SYCL_DEVICE_DESCRIPTOR_TEST(platform)
SYCL_DEVICE_DESCRIPTOR_TEST(name)
SYCL_DEVICE_DESCRIPTOR_TEST(vendor)
SYCL_DEVICE_DESCRIPTOR_TEST(driver_version)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(profile)
SYCL_DEVICE_DESCRIPTOR_TEST(version)
SYCL_DEVICE_DESCRIPTOR_TEST(backend_version)
SYCL_DEVICE_DESCRIPTOR_TEST(aspects)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(extensions)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(printf_buffer_size)
SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST(preferred_interop_user_sync)
SYCL_DEVICE_DESCRIPTOR_TEST(parent_device)
SYCL_DEVICE_DESCRIPTOR_TEST(partition_max_sub_devices)
SYCL_DEVICE_DESCRIPTOR_TEST(partition_properties)
SYCL_DEVICE_DESCRIPTOR_TEST(partition_affinity_domains)
SYCL_DEVICE_DESCRIPTOR_TEST(partition_type_property)
SYCL_DEVICE_DESCRIPTOR_TEST(partition_type_affinity_domain)

#undef SYCL_DEPRECATED_DEVICE_DESCRIPTOR_TEST
#undef SYCL_DEVICE_DESCRIPTOR_TEST

TEST_CASE("the device header defines the aspect enum",
          "[khr_split_headers][device]") {
  STATIC_REQUIRE(std::is_enum_v<sycl::aspect>);
}

#define SYCL_DEVICE_ENUM_TEST(enum_name)                         \
  TEST_CASE("the device header defines the " #enum_name " enum", \
            "[khr_split_headers][device]") {                     \
    STATIC_REQUIRE(std::is_enum_v<sycl::info::enum_name>);       \
  }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
#define SYCL_DEPRECATED_DEVICE_ENUM_TEST(enum_name) \
  SYCL_DEVICE_ENUM_TEST(enum_name)
#else
#define SYCL_DEPRECATED_DEVICE_ENUM_TEST(enum_name)
#endif

SYCL_DEVICE_ENUM_TEST(device_type)
SYCL_DEVICE_ENUM_TEST(partition_property)
SYCL_DEVICE_ENUM_TEST(partition_affinity_domain)
SYCL_DEVICE_ENUM_TEST(fp_config)
SYCL_DEVICE_ENUM_TEST(local_mem_type)
SYCL_DEVICE_ENUM_TEST(global_mem_cache_type)
SYCL_DEPRECATED_DEVICE_ENUM_TEST(execution_capability)

#undef SYCL_DEPRECATED_DEVICE_ENUM_TEST
#undef SYCL_DEVICE_ENUM_TEST

}  // namespace khr_split_headers::tests
