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

            /** check copy constructor
            */
            {
                cl::sycl::handler_event handlerEventA = queue.submit( [&]( cl::sycl::handler &cgh )
                {
                });

                cl::sycl::handler_event handlerEventB( handlerEventA );
            }

            /** check assignment operator
            */
            {
                cl::sycl::handler_event handlerEventA = queue.submit( [&]( cl::sycl::handler &cgh )
                {
                });

                cl::sycl::handler_event handlerEventB = handlerEventA;
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
