/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME       opencl_interop_constructors

namespace sycl_cts
{

/** check that we can instantiate a sycl platform class
 */
class TEST_NAME
    : public sycl_cts::util::test_base_opencl
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
            cl::sycl::platform  platform(m_cl_platform_id);
            cl::sycl::device    device(m_cl_device);
            cl::sycl::context   context(m_cl_context);
#if ENABLE_FULL_TEST
            cl::sycl::queue     queue(m_cl_command_queue);
#endif

            // Construct the vector with required values then pass
            // the address of its first element in to the ctor
            VECTOR_CLASS<cl_context_properties> properties;
            properties.push_back(CL_CONTEXT_PLATFORM);
            properties.push_back((cl_context_properties)m_cl_platform_id);
            properties.push_back(0);

            // construct the cts default selector
            cts_selector selector;
            VECTOR_CLASS<cl::sycl::device> devices;
            devices.push_back(device);

            cl::sycl::context ctxt_ds  (selector, &(properties[0]));
            cl::sycl::context ctxt_dev (device,   &(properties[0]));
            cl::sycl::context ctxt_plat(platform, &(properties[0]));
            cl::sycl::context ctxt_list(devices,  &(properties[0]));
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

}; // sycl_cts
