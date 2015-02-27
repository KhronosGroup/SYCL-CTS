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

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cl::sycl::event myEvent;

            auto evt = myEvent.get();
            if ( typeid( evt ) != typeid( cl_event ) )
            {
                FAIL( log,
                    "cl::sycl::event::get() does not "
                    "return cl_event" );
            }

            auto events = myEvent.get_wait_list();
            if (typeid(events) != typeid(VECTOR_CLASS<cl::sycl::event>))
            {
                FAIL(log,
                    "cl::sycl::event::get_wait_list() does not "
                    "return VECTOR_CLASS<cl::sycl::event>");
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace event_api__ */
