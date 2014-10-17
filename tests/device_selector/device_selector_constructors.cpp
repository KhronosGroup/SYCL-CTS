/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME       device_selector_constructors

namespace sycl_cts
{

/** check that we can instantiate various device selectors
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

    /** execute the test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {
        try
        {
            std::unique_ptr<cl::sycl::platform> plat;

            cl::sycl::default_selector  ds;
            cl::sycl::gpu_selector      gs;
            cl::sycl::cpu_selector      cs;
            cl::sycl::host_selector     hs;
#if ENABLE_FULL_TEST
            plat = new cl::sycl::platform(ds);
            plat = new cl::sycl::platform(gs);
            plat = new cl::sycl::platform(cs);
            plat = new cl::sycl::platform(hs);
#endif
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "" );
        }
    }
};

// register this test with the test_collection
static util::test_proxy<TEST_NAME> proxy;

} // namespace sycl_cts
