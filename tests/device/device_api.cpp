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
        set_test_info( out, TOSTRING( TEST_NAME ) );
    }

    /** execute the test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            // mark pass, which will be overwritten by a fail
            log.pass( );

            cl::sycl::device device;

            auto ret = device.is_host( );
            // validate return type
            if ( typeid( ret ) != typeid( bool ) )
            {
                log.fail( "cl::sycl::device::is_host( ) does not return bool type" );
            }

            auto devid = device.get( );
            // validate return type
            if ( typeid( devid ) != typeid( cl_device_id ) )
            {
                log.fail( "cl::sycl::device::get( ) does not return cl_device_id type" );
            }

            auto hext = device.has_extension( "" );
            // validate return type
            if ( typeid( hext ) != typeid( bool ) )
            {
                log.fail( "cl::sycl::device::has_extension( ) does not return bool type" );
            }

            // get a list of all devices
            VECTOR_CLASS<cl::sycl::device> devices = device.get_devices(CL_DEVICE_TYPE_ALL);

            for (int i = 0; i < devices.size(); i++)
            {
                cl::sycl::device & dev = devices[i];
            }
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            log.fail( );
        }
    }

};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
