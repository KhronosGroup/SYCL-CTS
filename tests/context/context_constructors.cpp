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
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            cl::sycl::context ctxt;
            cl::sycl::context ctxt_copy(ctxt);
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "" );
        }
    }

};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
