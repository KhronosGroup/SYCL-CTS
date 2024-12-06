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

TEST_CASE("device info", "[device]") {
  /** check info::device_type
   */
  check_enum_class_value(sycl::info::device_type::cpu);
  check_enum_class_value(sycl::info::device_type::gpu);
  check_enum_class_value(sycl::info::device_type::accelerator);
  check_enum_class_value(sycl::info::device_type::custom);
  check_enum_class_value(sycl::info::device_type::automatic);
  check_enum_class_value(sycl::info::device_type::host);
  check_enum_class_value(sycl::info::device_type::all);

  /** check info::partition_property
   */
  check_enum_class_value(sycl::info::partition_property::no_partition);
  check_enum_class_value(sycl::info::partition_property::partition_equally);
  check_enum_class_value(sycl::info::partition_property::partition_by_counts);
  check_enum_class_value(
      sycl::info::partition_property::partition_by_affinity_domain);

  /** check info::affinity_domain
   */
  check_enum_class_value(sycl::info::partition_affinity_domain::not_applicable);
  check_enum_class_value(sycl::info::partition_affinity_domain::numa);
  check_enum_class_value(sycl::info::partition_affinity_domain::L1_cache);
  check_enum_class_value(sycl::info::partition_affinity_domain::L2_cache);
  check_enum_class_value(sycl::info::partition_affinity_domain::L3_cache);
  check_enum_class_value(sycl::info::partition_affinity_domain::L4_cache);
  check_enum_class_value(
      sycl::info::partition_affinity_domain::next_partitionable);

  /** check info::local_mem_type
   */
  check_enum_class_value(sycl::info::local_mem_type::none);
  check_enum_class_value(sycl::info::local_mem_type::local);
  check_enum_class_value(sycl::info::local_mem_type::global);

  /** check info::fp_config
   */
  check_enum_class_value(sycl::info::fp_config::denorm);
  check_enum_class_value(sycl::info::fp_config::inf_nan);
  check_enum_class_value(sycl::info::fp_config::round_to_nearest);
  check_enum_class_value(sycl::info::fp_config::round_to_zero);
  check_enum_class_value(sycl::info::fp_config::round_to_inf);
  check_enum_class_value(sycl::info::fp_config::fma);
  check_enum_class_value(sycl::info::fp_config::correctly_rounded_divide_sqrt);
  check_enum_class_value(sycl::info::fp_config::soft_float);

  /** check global_mem_cache_type
   */
  check_enum_class_value(sycl::info::global_mem_cache_type::none);
  check_enum_class_value(sycl::info::global_mem_cache_type::read_only);
  check_enum_class_value(sycl::info::global_mem_cache_type::read_write);

  /** check execution_capability
   */
  check_enum_class_value(sycl::info::execution_capability::exec_kernel);
  check_enum_class_value(sycl::info::execution_capability::exec_native_kernel);

  /** check get_info parameters
   */
  // FIXME: Reenable when struct information descriptors are implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
  {
    auto dev = sycl_cts::util::get_cts_object::device(cts_selector);
    check_get_info_param<sycl::info::device::device_type,
                         sycl::info::device_type>(dev);
    check_get_info_param<sycl::info::device::vendor_id, uint32_t>(dev);
    check_get_info_param<sycl::info::device::max_compute_units, uint32_t>(dev);
    check_get_info_param<sycl::info::device::max_work_item_dimensions,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::max_work_item_sizes<1>,
                         sycl::range<1>>(dev);
    check_get_info_param<sycl::info::device::max_work_item_sizes<2>,
                         sycl::range<2>>(dev);
    check_get_info_param<sycl::info::device::max_work_item_sizes<3>,
                         sycl::range<3>>(dev);
    check_get_info_param<sycl::info::device::max_work_group_size, size_t>(dev);
    check_get_info_param<sycl::info::device::max_num_sub_groups, uint32_t>(dev);
    CHECK((dev.get_info<sycl::info::device::max_num_sub_groups>() != 0));
    check_get_info_param<sycl::info::device::sub_group_sizes,
                         std::vector<size_t>>(dev);
    check_get_info_param<sycl::info::device::preferred_vector_width_char,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::preferred_vector_width_short,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::preferred_vector_width_int,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::preferred_vector_width_long,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::preferred_vector_width_float,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::preferred_vector_width_double,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::preferred_vector_width_half,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::native_vector_width_char,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::native_vector_width_short,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::native_vector_width_int, uint32_t>(
        dev);
    check_get_info_param<sycl::info::device::native_vector_width_long,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::native_vector_width_float,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::native_vector_width_double,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::native_vector_width_half,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::max_clock_frequency, uint32_t>(
        dev);
    check_get_info_param<sycl::info::device::address_bits, uint32_t>(dev);
    check_get_info_param<sycl::info::device::max_mem_alloc_size, uint64_t>(dev);
    check_get_info_param<sycl::info::device::image_support, bool>(dev);
    check_get_info_param<sycl::info::device::max_read_image_args, uint32_t>(
        dev);
    check_get_info_param<sycl::info::device::max_write_image_args, uint32_t>(
        dev);
    check_get_info_param<sycl::info::device::image2d_max_height, size_t>(dev);
    check_get_info_param<sycl::info::device::image2d_max_width, size_t>(dev);
    check_get_info_param<sycl::info::device::image3d_max_height, size_t>(dev);
    check_get_info_param<sycl::info::device::image3d_max_width, size_t>(dev);
    check_get_info_param<sycl::info::device::image3d_max_depth, size_t>(dev);
    check_get_info_param<sycl::info::device::image_max_buffer_size, size_t>(
        dev);
    check_get_info_param<sycl::info::device::max_samplers, uint32_t>(dev);
    check_get_info_param<sycl::info::device::max_parameter_size, size_t>(dev);
    check_get_info_param<sycl::info::device::mem_base_addr_align, uint32_t>(
        dev);
    check_get_info_param<sycl::info::device::half_fp_config,
                         std::vector<sycl::info::fp_config>>(dev);
    check_get_info_param<sycl::info::device::single_fp_config,
                         std::vector<sycl::info::fp_config>>(dev);
    check_get_info_param<sycl::info::device::double_fp_config,
                         std::vector<sycl::info::fp_config>>(dev);
    check_get_info_param<sycl::info::device::global_mem_cache_type,
                         sycl::info::global_mem_cache_type>(dev);
    check_get_info_param<sycl::info::device::global_mem_cache_line_size,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::global_mem_cache_size, uint64_t>(
        dev);
    check_get_info_param<sycl::info::device::global_mem_size, uint64_t>(dev);
    check_get_info_param<sycl::info::device::max_constant_buffer_size,
                         uint64_t>(dev);
    check_get_info_param<sycl::info::device::max_constant_args, uint32_t>(dev);
    check_get_info_param<sycl::info::device::local_mem_type,
                         sycl::info::local_mem_type>(dev);
    check_get_info_param<sycl::info::device::local_mem_size, uint64_t>(dev);
    check_get_info_param<sycl::info::device::error_correction_support, bool>(
        dev);
    check_get_info_param<sycl::info::device::host_unified_memory, bool>(dev);
    {
      check_get_info_param<sycl::info::device::atomic_memory_order_capabilities,
                           std::vector<sycl::memory_order>>(dev);
      std::vector<sycl::memory_order> capabilities =
          dev.get_info<sycl::info::device::atomic_memory_order_capabilities>();
      CHECK(check_contains(capabilities, sycl::memory_order::relaxed));
    }
    {
      check_get_info_param<sycl::info::device::atomic_fence_order_capabilities,
                           std::vector<sycl::memory_order>>(dev);
      std::vector<sycl::memory_order> capabilities =
          dev.get_info<sycl::info::device::atomic_fence_order_capabilities>();
      CHECK(check_contains(capabilities, sycl::memory_order::relaxed));
      CHECK(check_contains(capabilities, sycl::memory_order::acquire));
      CHECK(check_contains(capabilities, sycl::memory_order::release));
      CHECK(check_contains(capabilities, sycl::memory_order::acq_rel));
    }
    {
      check_get_info_param<sycl::info::device::atomic_memory_scope_capabilities,
                           std::vector<sycl::memory_scope>>(dev);
      std::vector<sycl::memory_scope> capabilities =
          dev.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
      CHECK(check_contains(capabilities, sycl::memory_scope::work_group));
    }
    {
      check_get_info_param<sycl::info::device::atomic_fence_scope_capabilities,
                           std::vector<sycl::memory_scope>>(dev);
      std::vector<sycl::memory_scope> capabilities =
          dev.get_info<sycl::info::device::atomic_fence_scope_capabilities>();
      CHECK(check_contains(capabilities, sycl::memory_scope::work_group));
    }
    check_get_info_param<sycl::info::device::profiling_timer_resolution,
                         size_t>(dev);
    check_get_info_param<sycl::info::device::is_endian_little, bool>(dev);
    check_get_info_param<sycl::info::device::is_available, bool>(dev);
    check_get_info_param<sycl::info::device::is_compiler_available, bool>(dev);
    check_get_info_param<sycl::info::device::is_linker_available, bool>(dev);
    check_get_info_param<sycl::info::device::execution_capabilities,
                         std::vector<sycl::info::execution_capability>>(dev);
    check_get_info_param<sycl::info::device::queue_profiling, bool>(dev);
    check_get_info_param<sycl::info::device::built_in_kernel_ids,
                         std::vector<sycl::kernel_id>>(dev);
    check_get_info_param<sycl::info::device::built_in_kernels,
                         std::vector<std::string>>(dev);
    check_get_info_param<sycl::info::device::platform, sycl::platform>(dev);
    check_get_info_param<sycl::info::device::name, std::string>(dev);
    check_get_info_param<sycl::info::device::vendor, std::string>(dev);
    check_get_info_param<sycl::info::device::driver_version, std::string>(dev);
    check_get_info_param_backend_specific<sycl::info::device::profile,
                                          std::string, sycl::backend::opencl>(
        dev);
    check_get_info_param<sycl::info::device::version, std::string>(dev);
    check_get_info_param<sycl::info::device::backend_version, std::string>(dev);

    check_get_info_param<sycl::info::device::aspects,
                         std::vector<sycl::aspect>>(dev);

    check_get_info_param<sycl::info::device::extensions,
                         std::vector<std::string>>(dev);
    check_get_info_param<sycl::info::device::printf_buffer_size, size_t>(dev);
    check_get_info_param_backend_specific<
        sycl::info::device::preferred_interop_user_sync, bool,
        sycl::backend::opencl>(dev);
    auto SupportedProperties =
        dev.get_info<sycl::info::device::partition_properties>();
    if (std::find(SupportedProperties.begin(), SupportedProperties.end(),
                  sycl::info::partition_property::partition_equally) !=
        SupportedProperties.end()) {
      auto sub_device_partition_equal = dev.create_sub_devices<
          sycl::info::partition_property::partition_equally>(1);

      check_get_info_param<sycl::info::device::parent_device, sycl::device>(
          sub_device_partition_equal[0]);
    }
    check_get_info_param<sycl::info::device::partition_max_sub_devices,
                         uint32_t>(dev);
    check_get_info_param<sycl::info::device::partition_properties,
                         std::vector<sycl::info::partition_property>>(dev);
    check_get_info_param<sycl::info::device::partition_affinity_domains,
                         std::vector<sycl::info::partition_affinity_domain>>(
        dev);
    check_get_info_param<sycl::info::device::partition_type_property,
                         sycl::info::partition_property>(dev);
    check_get_info_param<sycl::info::device::partition_type_affinity_domain,
                         sycl::info::partition_affinity_domain>(dev);
  }
#endif
}
