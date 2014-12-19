/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_constructors

namespace opencl_interop_constructors__
{
using namespace sycl_cts;
template <typename T> using function_class = std::function<T>;

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
            cl_int error = CL_SUCCESS;

            cl::sycl::platform platform( m_cl_platform_id );
            cl::sycl::device device( m_cl_device );
            cl::sycl::context context( m_cl_context );
            cl::sycl::queue queue( m_cl_command_queue );

            /* create a cl_context_properties array */
            cl_context_properties ctext_properties[] =
            {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)m_cl_platform_id,
                0
            };

            /* construct the cts default selector */
            cts_selector selector;

            /* create a vector of devices */
            VECTOR_CLASS<cl::sycl::device> devices;
            devices.push_back( device );

            cl::sycl::context ctxt_ds  ( selector, ctext_properties );
            cl::sycl::context ctxt_dev ( device,   ctext_properties );
            cl::sycl::context ctxt_plat( platform, ctext_properties );
            cl::sycl::context ctxt_list( devices,  ctext_properties );

            // Bitfield specifying out-of-order execution and profiling
            cl_command_queue_properties queue_properties =
                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE &
                CL_QUEUE_PROFILING_ENABLE;

            function_class<void(cl::sycl::exception_list)> fn =
            [&] (cl::sycl::exception_list l)
            {
                if (l.size() > 1)
                    FAIL(log, "Exception thrown during execution of kernel");
            };

            cl::sycl::queue q_props( context, device, queue_properties );
            cl::sycl::queue q_handler( context, device, queue_properties, fn );
            cl::sycl::queue q_copy( m_cl_command_queue, fn );

            cl_bool normalise = 0;

            cl_sampler cl_s = clCreateSampler(
                m_cl_context,
                normalise,
                CL_ADDRESS_REPEAT,
                CL_FILTER_LINEAR,
                &error);
            CHECK_CL_SUCCESS( log, error );

            cl::sycl::sampler s(cl_s);

            error = clReleaseSampler(cl_s);
            CHECK_CL_SUCCESS( log, error );
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_as_functor */
