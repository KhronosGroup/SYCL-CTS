/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_get_info

namespace device_getinfo__
{
using namespace sycl_cts;

/** test cl::sycl::device initialization
*/
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
    *  @param info, test_base::info structure as output
    */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
    *  @param log, test transcript logging class
    */
    virtual void run( util::logger &log )
    {
        try
        {
            cl::sycl::device dev;

            if ( dev.is_host() )
                return;

            STRING_CLASS info_str;
            cl_uint info_uint;
            cl_bool info_bool;
            cl_ulong info_long;
            size_t info_size_t;
            cl_device_id info_dvid;
            cl_platform_id info_plid;

            cl_device_fp_config fpconfig;
            cl_device_exec_capabilities exec_caps;
            cl_device_mem_cache_type memctype;
            cl_device_local_mem_type lmemtype;
            cl_device_affinity_domain aff_domain;
            cl_command_queue_properties queue_props;
            cl_device_type dev_type;

            VECTOR_CLASS<size_t> info_vect_size_t;
            VECTOR_CLASS<cl_device_partition_property> info_vect_dpp;

            info_uint = dev.get_info<CL_DEVICE_ADDRESS_BITS>();
            info_bool = dev.get_info<CL_DEVICE_AVAILABLE>();
            info_str = dev.get_info<CL_DEVICE_BUILT_IN_KERNELS>();
            info_bool = dev.get_info<CL_DEVICE_COMPILER_AVAILABLE>();
            fpconfig = dev.get_info<CL_DEVICE_DOUBLE_FP_CONFIG>();
            info_bool = dev.get_info<CL_DEVICE_ENDIAN_LITTLE>();
            info_bool = dev.get_info<CL_DEVICE_ERROR_CORRECTION_SUPPORT>();
            exec_caps = dev.get_info<CL_DEVICE_EXECUTION_CAPABILITIES>();
            info_str = dev.get_info<CL_DEVICE_EXTENSIONS>();
            info_long = dev.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
            memctype = dev.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>();
            info_uint = dev.get_info<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
            info_long = dev.get_info<CL_DEVICE_GLOBAL_MEM_SIZE>();

            fpconfig = dev.get_info<CL_DEVICE_HALF_FP_CONFIG>();
            info_bool = dev.get_info<CL_DEVICE_HOST_UNIFIED_MEMORY>();
            info_bool = dev.get_info<CL_DEVICE_IMAGE_SUPPORT>();
            info_size_t = dev.get_info<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
            info_size_t = dev.get_info<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
            info_size_t = dev.get_info<CL_DEVICE_IMAGE3D_MAX_DEPTH>();
            info_size_t = dev.get_info<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
            info_size_t = dev.get_info<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
            info_size_t = dev.get_info<CL_DEVICE_IMAGE_MAX_BUFFER_SIZE>();
            info_size_t = dev.get_info<CL_DEVICE_IMAGE_MAX_ARRAY_SIZE>();
            info_bool = dev.get_info<CL_DEVICE_LINKER_AVAILABLE>();
            info_long = dev.get_info<CL_DEVICE_LOCAL_MEM_SIZE>();
            lmemtype = dev.get_info<CL_DEVICE_LOCAL_MEM_TYPE>();
            info_uint = dev.get_info<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
            info_uint = dev.get_info<CL_DEVICE_MAX_COMPUTE_UNITS>();
            info_uint = dev.get_info<CL_DEVICE_MAX_CONSTANT_ARGS>();
            info_long = dev.get_info<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
            info_long = dev.get_info<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            info_size_t = dev.get_info<CL_DEVICE_MAX_PARAMETER_SIZE>();
            info_uint = dev.get_info<CL_DEVICE_MAX_READ_IMAGE_ARGS>();
            info_uint = dev.get_info<CL_DEVICE_MAX_SAMPLERS>();
            info_size_t = dev.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            info_uint = dev.get_info<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
            info_vect_size_t = dev.get_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            info_uint = dev.get_info<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>();
            info_uint = dev.get_info<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();

            info_uint = dev.get_info<CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE>();

            info_str = dev.get_info<CL_DEVICE_NAME>();
            info_uint = dev.get_info<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR>();
            info_uint = dev.get_info<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>();
            info_uint = dev.get_info<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>();
            info_uint = dev.get_info<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG>();
            info_uint = dev.get_info<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>();
            info_uint = dev.get_info<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
            info_uint = dev.get_info<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>();
            info_str = dev.get_info<CL_DEVICE_OPENCL_C_VERSION>();
            info_dvid = dev.get_info<CL_DEVICE_PARENT_DEVICE>();
            info_uint = dev.get_info<CL_DEVICE_PARTITION_MAX_SUB_DEVICES>();

            info_vect_dpp = dev.get_info<CL_DEVICE_PARTITION_PROPERTIES>();
            aff_domain = dev.get_info<CL_DEVICE_PARTITION_AFFINITY_DOMAIN>();
            info_vect_dpp = dev.get_info<CL_DEVICE_PARTITION_TYPE>();
            info_plid = dev.get_info<CL_DEVICE_PLATFORM>();
            info_uint = dev.get_info<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>();
            info_uint = dev.get_info<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>();
            info_uint = dev.get_info<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>();
            info_uint = dev.get_info<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>();
            info_uint = dev.get_info<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
            info_uint = dev.get_info<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
            info_uint = dev.get_info<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>();

            info_size_t = dev.get_info<CL_DEVICE_PRINTF_BUFFER_SIZE>();
            info_bool = dev.get_info<CL_DEVICE_PREFERRED_INTEROP_USER_SYNC>();
            info_str = dev.get_info<CL_DEVICE_PROFILE>();
            info_size_t = dev.get_info<CL_DEVICE_PROFILING_TIMER_RESOLUTION>();
            queue_props = dev.get_info<CL_DEVICE_QUEUE_PROPERTIES>();
            info_uint = dev.get_info<CL_DEVICE_REFERENCE_COUNT>();
            fpconfig = dev.get_info<CL_DEVICE_SINGLE_FP_CONFIG>();
            dev_type = dev.get_info<CL_DEVICE_TYPE>();
            info_str = dev.get_info<CL_DEVICE_VENDOR>();
            info_uint = dev.get_info<CL_DEVICE_VENDOR_ID>();
            info_str = dev.get_info<CL_DEVICE_VERSION>();
            info_str = dev.get_info<CL_DRIVER_VERSION>();
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_getinfo__ */
