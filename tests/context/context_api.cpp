/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME context_api

// conformance test suite namespace
namespace sycl_cts
{

/** 
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
            log.pass( );

            // construct the cts default selector
            cts_selector selector;
            cl::sycl::context context( selector );

            auto cxt = context.get( );
            if ( typeid( cxt ) != typeid( cl_context ) )
                log.fail( "cl::sycl::context::get() does not return cl_context" );

            auto isHost = context.is_host( );
            if ( typeid( isHost ) != typeid( bool ) )
                log.fail( "cl::sycl::context::is_host() does not return bool" );

            // get a list of devices
            VECTOR_CLASS<cl::sycl::device> devices = context.get_devices();

            // loop over all devices
            for (int i = 0; i < devices.size(); i++)
            {
                cl::sycl::device &dev = devices[i];
            }
        }
        catch (cl::sycl::sycl_error e)
        {
            log_exception(log, e);
            log.fail();
        }
    }

};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
