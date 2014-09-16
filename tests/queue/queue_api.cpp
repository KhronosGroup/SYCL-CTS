/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME queue_api

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
        set_test_info( out, TOSTRING( TEST_NAME ));
    }

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            // set the test to pass until overwritten when it fails
            log.pass( );
            
            // construct the cts default selector
            cts_selector selector;

            // construct command queue
            cl::sycl::queue myQueue( selector );

            auto cxt = myQueue.get_context( );
            if (typeid( cxt ) != typeid( cl::sycl::context ))
            {
                log.fail( "cl::sycl::queue::get_context() does not return cl::sycl::context" );
            }

            auto dev = myQueue.get_device( );
            if (typeid( dev ) != typeid( cl::sycl::device ))
            {
                log.fail( "cl::sycl::queue::get_device() does not return cl::sycl::device" );
            }
            
            auto host = myQueue.is_host( );
            if (typeid( host ) != typeid( bool ))
            {
                log.fail( "cl::sycl::queue::is_host() does not return bool" );
            }

            auto cmdq = myQueue.get( );
            if (typeid( cmdq ) != typeid( cl_command_queue ))
            {
                log.fail( "cl::sycl::queue::get() does not return cl_command_queue" );
            }

        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            log.fail( );
        }
    }
};

// register this test with the test_collection
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
