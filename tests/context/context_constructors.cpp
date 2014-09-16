/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME context_constructors

namespace sycl_cts
{

/** check we can construct a SYCL context
 */
class TEST_NAME
    : public util::test_base
{
    
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
            log.pass();

            cl::sycl::context cxt1;

            // construct the cts default selector
            cts_selector selector;
            cl::sycl::context cxt2(selector);
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
