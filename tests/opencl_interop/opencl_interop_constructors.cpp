/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME opencl_interop_constructors

namespace sycl_cts
{

/** check that we can instantiate a sycl platform class
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl
{
public:
    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {
        try
        {
            cl::sycl::platform platform( m_cl_platform_id );
            cl::sycl::device device( m_cl_device );
            cl::sycl::context context( m_cl_context );
            cl::sycl::queue queue( m_cl_command_queue );

            /* create a cl_context_properties array */
            cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)m_cl_platform_id, 0 };

            /* construct the cts default selector */
            cts_selector selector;

            /* create a vector of devices */
            VECTOR_CLASS<cl::sycl::device> devices;
            devices.push_back( device );

            cl::sycl::context ctxt_ds( selector, properties );
            cl::sycl::context ctxt_dev( device, properties );
            cl::sycl::context ctxt_plat( platform, properties );
            cl::sycl::context ctxt_list( devices, properties );
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

};  // sycl_cts
