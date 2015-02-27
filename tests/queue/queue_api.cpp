/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_api

namespace queue_api__
{
using namespace sycl_cts;

/** tests the api for cl::sycl::queue
 */
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cts_selector selector;
            cl::sycl::queue queue( selector );

            /** check is_host() method
            */
            auto isHost = queue.is_host();
            if ( typeid( isHost ) != typeid(bool))
            {
                FAIL( log, "is_host() does not return bool" );
            }

            /** check get_context() method
            */
            auto context = queue.get_context();
            if ( typeid( context ) != typeid( cl::sycl::context ))
            {
                FAIL( log, "is_context() does not return context" );
            }

            /** check get_device() method
            */
            auto device = queue.get_device();
            if ( typeid( device ) != typeid( cl::sycl::device ))
            {
                FAIL( log, "get_device() does not return context" );
            }

            /** check submit(command_group_scope) method
            */
            auto handlerEvent = queue.submit( [&]( cl::sycl::handler &handler )
            {
            } );
            if ( typeid( handlerEvent ) != typeid( cl::sycl::handler_event ) )
            {
                FAIL( log, "submit(command_group_scope) does not return handler_event" );
            }

            /** check submit(command_group_scope, queue) method
            */
            cl::sycl::queue secondaryQueue( selector );
            auto handlerEvent = queue.submit( [&]( cl::sycl::handler &handler )
            {
            }, secondaryQueue );
            if ( typeid( handlerEvent ) != typeid( cl::sycl::handler_event ) )
            {
                FAIL( log, "submit(command_group_scope, queue) does not return handler_event" );
            }

            /** check wait() method
            */
            queue.wait();

            /** check wait_and_throw() method
            */
            queue.wait_and_throw();

            /** check throw_asynchronous() method
            */
            queue.throw_asynchronous();
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "a sycl exception was caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace queue_api__ */
