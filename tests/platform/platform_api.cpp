/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_api

namespace platform_api__
{
using namespace sycl_cts;

/** check that we can instantiate a sycl platform class
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

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {
        try
        {
            cl::sycl::platform plat;

            /* Ask for the OpenCL platform associated with
             * the cl::sycl::platform
             */
            auto cl_plat = plat.get();
            if ( typeid( cl_plat ) != typeid( cl_platform_id ) )
                FAIL( log,
                      "platform.get() does not return "
                      "cl_platform_id" );

            /* static method, should return all platforms available */
            auto platforms = cl::sycl::platform::get_platforms();
            if ( typeid( platforms ) != typeid(VECTOR_CLASS<cl::sycl::platform>))
                FAIL( log,
                      "get_platforms() does not return "
                      "VECTOR_CLASS<cl::sycl::platform>" );

            /* Static method, should return all devices across all platforms */
            auto devices = cl::sycl::platform::get_devices();
            if ( typeid( devices ) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log,
                      "get_devices() does not return "
                      "VECTOR_CLASS<cl::sycl::device>" );

            /* Static method should return all CPU devices on all platforms */
            auto cpu_devices = cl::sycl::platform::get_devices( CL_DEVICE_TYPE_CPU );
            if ( typeid( devices ) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log,
                      "get_devices(cl_device_type) does not return "
                      "VECTOR_CLASS<cl::sycl::device>" );

            /* Expect all devices on one platform */
            auto cl_devices = plat.get_devices();
            if ( typeid( cl_devices ) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log,
                      "get_devices(cl_device_type) does not return "
                      "VECTOR_CLASS<cl::sycl::device>" );

            /* ask for all GPU devices on a single platform */
            cl_devices = plat.get_devices( CL_DEVICE_TYPE_GPU );
            if ( typeid( cl_devices ) != typeid(VECTOR_CLASS<cl::sycl::device>))
                FAIL( log,
                      "get_devices(cl_device_type) does not return "
                      "VECTOR_CLASS<cl::sycl::device>" );

            /* Check for platform's support of extensions */
            STRING_CLASS ext( "cl_khr_icd" );
            auto fp64_plat = plat.has_extension( ext );
            if ( typeid( fp64_plat ) != typeid(bool))
                FAIL( log, "platform.has_extension() does not return bool" );

            /* Test presence of is_host function */
            auto is_host_plat = plat.is_host();
            if ( typeid( is_host_plat ) != typeid(bool))
                FAIL( log, "platform.is_host() does not return bool" );
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_as_functor__ */
