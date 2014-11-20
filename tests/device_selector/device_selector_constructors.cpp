/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME device_selector_constructors

namespace sycl_cts
{

/** check that we can instantiate various device selectors
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
            /* Predefined selectors */
            cl::sycl::default_selector ds;
            cl::sycl::gpu_selector gs;
            cl::sycl::cpu_selector cs;
            cl::sycl::host_selector hs;

            /* Create device from each selector */
            cl::sycl::device dev( ds );
            cl::sycl::device dev_gpu( gs );
            cl::sycl::device dev_cpu( cs );
            cl::sycl::device dev_host( hs );

            if ( !( dev_gpu.get_info<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU ) )
                FAIL( log, "dev_gpu is not a GPU device" );

            if ( !( dev_cpu.get_info<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU ) )
                FAIL( log, "dev_cpu is not a GPU device" );

            if ( !dev_host.is_host() )
                FAIL( log, "dev_host is not a host device" );

        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// register this test with the test_collection
static util::test_proxy<TEST_NAME> proxy;

}  // namespace sycl_cts
