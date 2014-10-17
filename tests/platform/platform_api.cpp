/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME platform_api

namespace sycl_cts
{

/** check that we can instantiate a sycl platform class
 */
class TEST_NAME
    : public util::test_base
{
public:
    
    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info & out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            cl::sycl::platform plat;

            auto cl_plat = plat.get();
            if(typeid(cl_plat) != typeid(cl_platform_id))
                FAIL( log, "platform.get() does not return " \
                "cl_platform_id");

            auto platforms = cl::sycl::platform::get_platforms();
            if(typeid(platforms) != typeid(VECTOR_CLASS<cl::sycl::platform>))
                FAIL( log, "get_plaforms() does not return " \
                "VECTOR_CLASS<cl::sycl::platform>");

#if ENABLE_FULL_TEST
            auto devices = cl::sycl::platform::get_devices();
            if(typeid(devices) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log, "get_devices() does not return " \
                "VECTOR_CLASS<cl::sycl::device>");

            auto cpu_devices =
                cl::sycl::platform::get_devices(CL_DEVICE_TYPE_CPU);
            if(typeid(devices) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log, "get_devices(cl_device_type) does not return " \
                "VECTOR_CLASS<cl::sycl::device>");
#endif

            auto cl_devices = plat.get_devices();
            if(typeid(cl_devices) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log, "get_devices(cl_device_type) does not return " \
                "VECTOR_CLASS<cl::sycl::device>");

            cl_devices = plat.get_devices(CL_DEVICE_TYPE_GPU);
            if(typeid(cl_devices) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log, "get_devices(cl_device_type) does not return " \
                "VECTOR_CLASS<cl::sycl::device>");

            STRING_CLASS ext("cl_khr_icd");
            auto fp64_plat = plat.has_extension(ext);
            if(typeid(fp64_plat) != typeid(bool))
                FAIL( log, "platform.has_extension() does not return bool");

            auto is_host_plat = plat.is_host();
            if(typeid(is_host_plat) != typeid(bool))
                FAIL( log, "platform.is_host() does not return bool");

            auto is_opencl_plat = plat.is_host();
            if(typeid(is_opencl_plat) != typeid(bool))
                FAIL( log, "platform.is_host() does not return bool");
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "" );
        }
    }

};

// register this test with the test_collection
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
