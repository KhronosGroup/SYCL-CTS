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
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            // construct the cts default selector
            cts_selector selector;

            // construct command queue
            cl::sycl::queue myQueue(selector);

            if (!myQueue.is_host())
            {
                auto cmdq = myQueue.get();
                if (typeid(cmdq) != typeid(cl_command_queue))
                {
                    FAIL(log, "cl::sycl::queue::get() does not return cl_command_queue");
                }
            }

            auto ctxt = myQueue.get_context( );
            if (typeid( ctxt ) != typeid( cl::sycl::context ))
            {
                FAIL( log, "cl::sycl::queue::get_context() does not return cl::sycl::context" );
            }

            auto dev = myQueue.get_device( );
            if (typeid( dev ) != typeid( cl::sycl::device ))
            {
                FAIL( log, "cl::sycl::queue::get_device() does not return cl::sycl::device" );
            }

#if ENABLE_FULL_TEST
            myQueue.wait();
            myQueue.wait_and_throw();
            myQueue.throw_asynchronous();
#endif

            auto host = myQueue.is_host( );
            if (typeid( host ) != typeid( bool ))
            {
                FAIL( log, "cl::sycl::queue::is_host() does not return bool" );
            }

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
