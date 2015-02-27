/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME handler_event_constructors

namespace handler_event_constructors__
{
using namespace sycl_cts;

/** tests the constructors for cl::sycl::handler_event
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

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cl::sycl::queue queue;

            {
                cl::sycl::handler_event handlerEvent = queue.submit( [&]( cl::sycl::handler &cgh )
                {
                });

                /** check get_kernel() method
                */
                auto kernelEvent = handlerEvent.get_kernel();
                if ( typeid( kernelEvent ) != typeid( cl::sycl::event ))
                {
                    FAIL( log, "cl::sycl::handler_event::get_kernel() does not return cl::sycl::event" );
                }

                /** check get_complete() method
                */
                auto completeEvent = handlerEvent.get_complete();
                if ( typeid( completeEvent ) != typeid( cl::sycl::event ))
                {
                    FAIL( log, "cl::sycl::handler_event::get_complete() does not return cl::sycl::event" );
                }

                /** check get_end() method
                */
                auto endEvent = handlerEvent.get_end();
                if ( typeid( endEvent ) != typeid( cl::sycl::event ))
                {
                    FAIL( log, "cl::sycl::handler_event::get_end() does not return cl::sycl::event" );
                }
            }
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

} /* namespace handler_event_constructors__ */
