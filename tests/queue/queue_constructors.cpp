/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME queue_constructors

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
        set_test_info( out, TOSTRING( TEST_NAME ) );
    }

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            log.pass();

            cl::sycl::queue myQueue;
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
