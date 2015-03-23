/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_info

namespace device_info__ {
using namespace sycl_cts;

/** tests the info for cl::sycl::device
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  */
  virtual void run(util::logger &log) override {
    try {
      cts_selector selector;
      cl::sycl::device device(selector);

      /** check types
      */
      using deviceFPConfig = cl::sycl::info::device_fp_config;
      using deviceExecCapabilities = cl::sycl::info::device_exec_capabilities;
      using deviceExecCapabilities = cl::sycl::info::device_exec_capabilities;
      using devicePartitionProperty = cl::sycl::info::device_partition_property;
      using deviceAfinityDomain = cl::sycl::info::device_affinity_domain;
      using devicePartitionType = cl::sycl::info::device_partition_type;
      using localMemType = cl::sycl::info::local_mem_type;
      using globalMemCacheType = cl::sycl::info::global_mem_cache_type;
      using queueProperties = cl::sycl::info::queue_properties;
      using queueInfo = cl::sycl::info::queue;

      /** initialize return values
      */
      cl_uint info_uint;
      cl_bool info_bool;
      cl_ulong info_long;
      size_t info_size_t;
      cl::sycl::info::device_id info_dvid;
      cl::sycl::info::device_fp_config fpconfig;
      cl::sycl::info::device_exec_capabilities exec_caps;
      cl::sycl::info::device_mem_cache_type memctype;
      cl::sycl::info::device_local_mem_type lmemtype;
      cl::sycl::info::device_affinity_domain aff_domain;
      cl::sycl::info::command_queue_properties queue_props;
      cl::sycl::info::device_type dev_type;
      cl::sycl::string_class info_str;
      cl::sycl::vector_class<size_t> info_vect_size_t;
      cl::sycl::vector_class<cl::sycl::info::device::partition_property>
          info_vect_dpp;

      /** check device info parameters
      */
      info_uint = device.get_info<cl::sycl::info::device::address_bits>();
      info_bool = device.get_info<cl::sycl::info::device::available>();
      info_str = device.get_info<cl::sycl::info::device::built_in_kernels>();
      info_bool = device.get_info<cl::sycl::info::device::compiler_available>();
      fpconfig = device.get_info<cl::sycl::info::device::double_fp_config>();
      info_bool = device.get_info<cl::sycl::info::device::endian_little>();
      info_bool =
          device.get_info<cl::sycl::info::device::error_correction_support>();
      exec_caps =
          device.get_info<cl::sycl::info::device::execution_capabilities>();
      info_str = device.get_info<cl::sycl::info::device::extensions>();
      info_long =
          device.get_info<cl::sycl::info::device::global_mem_cache_size>();
      memctype =
          device.get_info<cl::sycl::info::device::global_mem_cache_type>();
      info_uint =
          device.get_info<cl::sycl::info::device::global_mem_cacheline_size>();
      info_long = device.get_info<cl::sycl::info::device::global_mem_size>();
      fpconfig = device.get_info<cl::sycl::info::device::half_fp_config>();
      info_bool =
          device.get_info<cl::sycl::info::device::host_unified_memory>();
      info_bool = device.get_info<cl::sycl::info::device::image_support>();
      info_size_t =
          device.get_info<cl::sycl::info::device::image2d_max_height>();
      info_size_t =
          device.get_info<cl::sycl::info::device::image2d_max_width>();
      info_size_t =
          device.get_info<cl::sycl::info::device::image3d_max_depth>();
      info_size_t =
          device.get_info<cl::sycl::info::device::image3d_max_height>();
      info_size_t =
          device.get_info<cl::sycl::info::device::image3d_max_width>();
      info_size_t =
          device.get_info<cl::sycl::info::device::image_max_buffer_size>();
      info_size_t =
          device.get_info<cl::sycl::info::device::image_max_array_size>();
      info_bool = device.get_info<cl::sycl::info::device::linker_available>();
      info_long = device.get_info<cl::sycl::info::device::local_mem_size>();
      lmemtype = device.get_info<cl::sycl::info::device::local_mem_type>();
      info_uint =
          device.get_info<cl::sycl::info::device::max_clock_frequency>();
      info_uint = device.get_info<cl::sycl::info::device::max_compute_units>();
      info_uint = device.get_info<cl::sycl::info::device::max_constant_args>();
      info_long =
          device.get_info<cl::sycl::info::device::max_constant_buffer_size>();
      info_long = device.get_info<cl::sycl::info::device::max_mem_alloc_size>();
      info_size_t =
          device.get_info<cl::sycl::info::device::max_parameter_size>();
      info_uint =
          device.get_info<cl::sycl::info::device::max_read_image_args>();
      info_uint = device.get_info<cl::sycl::info::device::max_samplers>();
      info_size_t =
          device.get_info<cl::sycl::info::device::max_work_group_size>();
      info_uint =
          device.get_info<cl::sycl::info::device::max_work_item_dimensions>();
      info_vect_size_t =
          device.get_info<cl::sycl::info::device::max_work_item_sizes>();
      info_uint =
          device.get_info<cl::sycl::info::device::max_write_image_args>();
      info_uint =
          device.get_info<cl::sycl::info::device::mem_base_addr_align>();
      info_uint =
          device.get_info<cl::sycl::info::device::min_data_type_align_size>();
      info_str = device.get_info<cl::sycl::info::device::name>();
      info_uint =
          device.get_info<cl::sycl::info::device::native_vector_width_char>();
      info_uint =
          device.get_info<cl::sycl::info::device::native_vector_width_short>();
      info_uint =
          device.get_info<cl::sycl::info::device::native_vector_width_int>();
      info_uint =
          device.get_info<cl::sycl::info::device::native_vector_width_long>();
      info_uint =
          device.get_info<cl::sycl::info::device::native_vector_width_float>();
      info_uint =
          device.get_info<cl::sycl::info::device::native_vector_width_double>();
      info_uint =
          device.get_info<cl::sycl::info::device::native_vector_width_half>();
      info_str = device.get_info<cl::sycl::info::device::opencl_c_version>();
      info_dvid = device.get_info<cl::sycl::info::device::parent_device>();
      info_uint =
          device.get_info<cl::sycl::info::device::partition_max_sub_devices>();
      info_vect_dpp =
          device.get_info<cl::sycl::info::device::partition_properties>();
      aff_domain =
          device.get_info<cl::sycl::info::device::partition_affinity_domain>();
      info_vect_dpp = device.get_info<cl::sycl::info::device::partition_type>();
      info_plid = device.get_info<cl::sycl::info::device::platform>();
      info_uint =
          device
              .get_info<cl::sycl::info::device::preferred_vector_width_char>();
      info_uint =
          device
              .get_info<cl::sycl::info::device::preferred_vector_width_short>();
      info_uint =
          device.get_info<cl::sycl::info::device::preferred_vector_width_int>();
      info_uint =
          device
              .get_info<cl::sycl::info::device::preferred_vector_width_long>();
      info_uint =
          device
              .get_info<cl::sycl::info::device::preferred_vector_width_float>();
      info_uint = device.get_info<
          cl::sycl::info::device::preferred_vector_width_double>();
      info_uint =
          device
              .get_info<cl::sycl::info::device::preferred_vector_width_half>();
      info_size_t =
          device.get_info<cl::sycl::info::device::printf_buffer_size>();
      info_bool =
          device
              .get_info<cl::sycl::info::device::preferred_interop_user_sync>();
      info_str = device.get_info<cl::sycl::info::device::profile>();
      info_size_t =
          device.get_info<cl::sycl::info::device::profiling_timer_resolution>();
      queue_props = device.get_info<cl::sycl::info::device::queue_properties>();
      info_uint = device.get_info<cl::sycl::info::device::reference_count>();
      fpconfig = device.get_info<cl::sycl::info::device::single_fp_config>();
      dev_type = device.get_info<cl::sycl::info::device::type>();
      info_str = device.get_info<cl::sycl::info::device::vendor>();
      info_uint = device.get_info<cl::sycl::info::device::vendor_id>();
      info_str = device.get_info<cl::sycl::info::device::version>();
      info_str = device.get_info<cl::sycl::info::driver_version>();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_info__ */
