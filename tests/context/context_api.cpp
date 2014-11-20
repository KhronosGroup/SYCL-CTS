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
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {
        try
        {
            cl::sycl::context context;

            auto cxt = context.get();
            if ( typeid( cxt ) != typeid( cl_context ) )
                FAIL( log,
                      "cl::sycl::context::get() does not "
                      "return cl_context" );

            auto isHost = context.is_host();
            if ( typeid( isHost ) != typeid(bool))
                FAIL( log,
                      "cl::sycl::context::is_host() does not "
                      "return bool" );
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

};  // sycl_cts
