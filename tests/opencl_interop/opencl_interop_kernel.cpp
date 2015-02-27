/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_kernel

namespace TEST_NAMESPACE
{
using namespace sycl_cts;

static const util::STRING kern =
R"(__kernel void test_kern(__global int * arg_one, __private int arg_two)
{
    ;
})";

/** Test for the SYCL buffer OpenCL interoperation
*/
class TEST_NAME : public sycl_cts::util::test_base_opencl
{
public:
    /** return information about this test
    */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
    */
    virtual void run( util::logger &log ) override
    {
        /* get the OpenCLHelper object */
        using sycl_cts::util::opencl_helper;
        using sycl_cts::util::get;
        opencl_helper &openclHelper = get<opencl_helper>();

        try
        {
            const size_t size = 32;
            int data[size] = { 0 };

            cl_int error = CL_SUCCESS;

            cl::sycl::queue q( m_cl_command_queue );

            cl_mem opencl_buffer =
                clCreateBuffer( m_cl_context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                size * sizeof( int ),
                data,
                &error );

            if ( !CHECK_CL_SUCCESS( log, error ) )
                return;

            cl_program prog = nullptr;
            cl_kernel cl_kern = nullptr;

            if ( !create_program( kern, prog, log ) )
            {
                log.note( "Unable to create test OpenCL program" );
            }

            if ( !create_kernel( prog, "test_kern", cl_kern, log ) )
            {
                log.note( "Unable to create test OpenCL kernel" );
            }

            cl::sycl::kernel k( cl_kern );

            cl::sycl::buffer<int, size> buffer( opencl_buffer, m_cl_command_queue, nullptr );

            q.submit( [&]( cl::sycl::handler & cgh )
            {
                auto acc = buffer.get_access<mode::read_write, target::cl_buffer>( cgh );

                handler.set_arg( 0, acc );
                handler.set_arg( 1, size );

                cgh.single_task( k )
            } );

            error = clReleaseMemObject( opencl_buffer );
            if ( !CHECK_CL_SUCCESS( log, error ) )
                return;

            error = clReleaseKernel( cl_kern );
            if ( !CHECK_CL_SUCCESS( log, error ) )
                return;

            error = clReleaseProgram( prog );
            if ( !CHECK_CL_SUCCESS( log, error ) )
                return;
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

} /* namespace opencl_interop_buffer__ */
