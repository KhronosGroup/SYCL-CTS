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

#define TEST_NAME device_info

namespace device_info__ {

using namespace sycl_cts;

/** tests the info for sycl::device
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    {
      /** check info::device_type
       */
      check_enum_class_value(sycl::info::device_type::cpu);
      check_enum_class_value(sycl::info::device_type::gpu);
      check_enum_class_value(sycl::info::device_type::accelerator);
      check_enum_class_value(sycl::info::device_type::custom);
      check_enum_class_value(sycl::info::device_type::automatic);
      check_enum_class_value(sycl::info::device_type::host);
      check_enum_class_value(sycl::info::device_type::all);

      /** check info::partion_property
       */
      check_enum_class_value(sycl::info::partition_property::no_partition);
      check_enum_class_value(
          sycl::info::partition_property::partition_equally);
      check_enum_class_value(
          sycl::info::partition_property::partition_by_counts);
      check_enum_class_value(
          sycl::info::partition_property::partition_by_affinity_domain);

      /** check info::affinity_domain
       */
      check_enum_class_value(
          sycl::info::partition_affinity_domain::not_applicable);
      check_enum_class_value(sycl::info::partition_affinity_domain::numa);
      check_enum_class_value(
          sycl::info::partition_affinity_domain::L1_cache);
      check_enum_class_value(
          sycl::info::partition_affinity_domain::L2_cache);
      check_enum_class_value(
          sycl::info::partition_affinity_domain::L3_cache);
      check_enum_class_value(
          sycl::info::partition_affinity_domain::L4_cache);
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
      check_enum_class_value(
          sycl::info::fp_config::correctly_rounded_divide_sqrt);
      check_enum_class_value(sycl::info::fp_config::soft_float);

      /** check global_mem_cache_type
       */
      check_enum_class_value(sycl::info::global_mem_cache_type::none);
      check_enum_class_value(sycl::info::global_mem_cache_type::read_only);
      check_enum_class_value(sycl::info::global_mem_cache_type::read_write);

      /** check execution_capability
       */
      check_enum_class_value(sycl::info::execution_capability::exec_kernel);
      check_enum_class_value(
          sycl::info::execution_capability::exec_native_kernel);

      /** check get_info parameters
       */
      // FIXME: Reenable when struct information descriptors are implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        check_get_info_param<sycl::info::device::device_type,
                             sycl::info::device_type>(log, dev);
        check_get_info_param<sycl::info::device::vendor_id, sycl::cl_uint>(log,
                                                                           dev);
        check_get_info_param<sycl::info::device::max_compute_units,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::max_work_item_dimensions,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::max_work_item_sizes<1>,
                             sycl::id<1>>(log, dev);
        check_get_info_param<sycl::info::device::max_work_item_sizes<2>,
                             sycl::id<2>>(log, dev);
        check_get_info_param<sycl::info::device::max_work_item_sizes<3>,
                             sycl::id<3>>(log, dev);
        check_get_info_param<sycl::info::device::max_work_group_size, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::preferred_vector_width_char,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::preferred_vector_width_short,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::preferred_vector_width_int,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::preferred_vector_width_long,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::preferred_vector_width_float,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::preferred_vector_width_double,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::preferred_vector_width_half,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::native_vector_width_char,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::native_vector_width_short,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::native_vector_width_int,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::native_vector_width_long,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::native_vector_width_float,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::native_vector_width_double,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::native_vector_width_half,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::max_clock_frequency,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::address_bits, sycl::cl_uint>(
            log, dev);
        check_get_info_param<sycl::info::device::max_mem_alloc_size,
                             sycl::cl_ulong>(log, dev);
        check_get_info_param<sycl::info::device::image_support, bool>(log, dev);
        check_get_info_param<sycl::info::device::max_read_image_args,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::max_write_image_args,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::image2d_max_height, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::image2d_max_width, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::image3d_max_height, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::image3d_max_width, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::image3d_max_depth, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::image_max_buffer_size, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::image_max_array_size, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::max_samplers, sycl::cl_uint>(
            log, dev);
        check_get_info_param<sycl::info::device::max_parameter_size, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::mem_base_addr_align,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::half_fp_config,
                             std::vector<sycl::info::fp_config>>(log, dev);
        check_get_info_param<sycl::info::device::single_fp_config,
                             std::vector<sycl::info::fp_config>>(log, dev);
        check_get_info_param<sycl::info::device::double_fp_config,
                             std::vector<sycl::info::fp_config>>(log, dev);
        check_get_info_param<sycl::info::device::global_mem_cache_type,
                             sycl::info::global_mem_cache_type>(log, dev);
        check_get_info_param<sycl::info::device::global_mem_cache_line_size,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::global_mem_cache_size,
                             sycl::cl_ulong>(log, dev);
        check_get_info_param<sycl::info::device::global_mem_size,
                             sycl::cl_ulong>(log, dev);
        check_get_info_param<sycl::info::device::max_constant_buffer_size,
                             sycl::cl_ulong>(log, dev);
        check_get_info_param<sycl::info::device::max_constant_args,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::local_mem_type,
                             sycl::info::local_mem_type>(log, dev);
        check_get_info_param<sycl::info::device::local_mem_size,
                             sycl::cl_ulong>(log, dev);
        check_get_info_param<sycl::info::device::error_correction_support,
                             bool>(log, dev);
        check_get_info_param<sycl::info::device::host_unified_memory, bool>(
            log, dev);
        check_get_info_param<sycl::info::device::profiling_timer_resolution,
                             size_t>(log, dev);
        check_get_info_param<sycl::info::device::is_endian_little, bool>(log,
                                                                         dev);
        check_get_info_param<sycl::info::device::is_available, bool>(log, dev);
        check_get_info_param<sycl::info::device::is_compiler_available, bool>(
            log, dev);
        check_get_info_param<sycl::info::device::is_linker_available, bool>(
            log, dev);
        check_get_info_param<sycl::info::device::execution_capabilities,
                             std::vector<sycl::info::execution_capability>>(
            log, dev);
        check_get_info_param<sycl::info::device::queue_profiling, bool>(log,
                                                                        dev);
        check_get_info_param<sycl::info::device::built_in_kernels,
                             std::vector<std::string>>(log, dev);
        check_get_info_param<sycl::info::device::platform, sycl::platform>(log,
                                                                           dev);
        check_get_info_param<sycl::info::device::name, std::string>(log, dev);
        check_get_info_param<sycl::info::device::vendor, std::string>(log, dev);
        check_get_info_param<sycl::info::device::driver_version, std::string>(
            log, dev);
        check_get_info_param<sycl::info::device::profile, std::string>(log,
                                                                       dev);
        check_get_info_param<sycl::info::device::version, std::string>(log,
                                                                       dev);
        check_get_info_param<sycl::info::device::opencl_c_version, std::string>(
            log, dev);
        check_get_info_param<sycl::info::device::extensions,
                             std::vector<std::string>>(log, dev);
        check_get_info_param<sycl::info::device::printf_buffer_size, size_t>(
            log, dev);
        check_get_info_param<sycl::info::device::preferred_interop_user_sync,
                             bool>(log, dev);
        auto SupportedProperties =
            dev.get_info<sycl::info::device::partition_properties>();
        if (std::find(SupportedProperties.begin(), SupportedProperties.end(),
                      sycl::info::partition_property::partition_equally) !=
            SupportedProperties.end()) {

          auto sub_device_partition_equal = dev.create_sub_devices<
              sycl::info::partition_property::partition_equally>(1);

          check_get_info_param<sycl::info::device::parent_device, sycl::device>(
              log, sub_device_partition_equal[0]);
        }
        check_get_info_param<sycl::info::device::partition_max_sub_devices,
                             sycl::cl_uint>(log, dev);
        check_get_info_param<sycl::info::device::partition_properties,
                             std::vector<sycl::info::partition_property>>(log,
                                                                          dev);
        check_get_info_param<
            sycl::info::device::partition_affinity_domains,
            std::vector<sycl::info::partition_affinity_domain>>(log, dev);
        check_get_info_param<sycl::info::device::partition_type_property,
                             sycl::info::partition_property>(log, dev);
        check_get_info_param<sycl::info::device::partition_type_affinity_domain,
                             sycl::info::partition_affinity_domain>(log, dev);
      }
#endif
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_info__ */
