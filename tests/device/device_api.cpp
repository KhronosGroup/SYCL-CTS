/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME device_api

namespace sycl_cts
{

/** test cl::sycl::device::is_host() return type
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

    /** execute the test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            cl::sycl::device device;

            auto devid = device.get();
            if (typeid(devid) != typeid(cl_device_id))
            {
                FAIL(log, "cl::sycl::device::get( ) does not" \
                    "return cl_device_id type");
            }

            auto platform = device.get_platform();
            if (typeid(platform) != typeid(cl::sycl::platform))
            {
                FAIL(log, "cl::sycl::device::get_platform does not" \
                    "return cl::sycl::platform");
            }

            auto ret = device.is_host( );
            if ( typeid( ret ) != typeid( bool ) )
            {
                FAIL( log, "cl::sycl::device::is_host( ) does not"\
                    "return bool type" );
            }

            auto hext = device.has_extension( "cl_khr_fp64" );
            if ( typeid( hext ) != typeid( bool ) )
            {
                FAIL( log, "cl::sycl::device::has_extension( ) does not" \
                    "return bool type" );
            }

#if ENABLE_FULL_TEST
            //create_sub_devices should go here: see redmine #4377
#endif

            auto devices = cl::sycl::device::get_devices();
            if (typeid(devices) != typeid(VECTOR_CLASS<cl::sycl::device>))
            {
                FAIL(log, "cl::sycl::device::get_devices does not" \
                    "return VECTOR_CLASS<cl::sycl::device>");
            }
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "" );
        }
    }

};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
