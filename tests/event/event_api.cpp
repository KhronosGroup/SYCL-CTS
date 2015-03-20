/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_api

namespace event_api__
{
using namespace sycl_cts;

/**
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
            cl::sycl::event event = queue.submit( [&]( cl::sycl::handler &handler )
                                                  {
                                                  } ).get_complete();

            auto evt = event.get();
            if ( typeid( evt ) != typeid( cl_event ) )
            {
                FAIL( log, "get() does not return cl_event" );
            }

            auto events = event.get_wait_list();
            if ( typeid( events ) != typeid(cl::sycl::vector_class<cl::sycl::event>))
            {
                FAIL( log, "get_wait_list() does not return vector_class<event>" );
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "a sycl exception was caught" );
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace event_api__ */
