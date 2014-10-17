/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME platform_constructors

namespace sycl_cts
{

/** check that we can instantiate a sycl platform class
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
            cl::sycl::platform p;

#if ENABLE_FULL_TEST
            cl::sycl::host_selector hs;
            cl::sycl::platform p_selector(ds);
#endif
            cl::sycl::platform p_copy(p);
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "Failed to construct platform object in platform_constructors" );
        }
    }

};

// register this test with the test_collection
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
