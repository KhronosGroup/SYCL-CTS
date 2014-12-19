/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_buffer

namespace opencl_interop_buffer__
{
using namespace sycl_cts;

/** Test for the SYCL buffer OpenCL interoperation
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
        /* get the OpenCLHelper object */
        using sycl_cts::util::opencl_helper;
        using sycl_cts::util::get;
        opencl_helper &openclHelper = get<opencl_helper>();

        try
        {
            const size_t size = 32;
            int data[size] = { 0 };

            cl_int error = CL_SUCCESS;

            cl_mem opencl_buffer =
                clCreateBuffer( m_cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof( int ), data, &error );
            if ( !CHECK_CL_SUCCESS( log, error ) )
                return;

            cl::sycl::buffer<int, size> buffer( opencl_buffer, m_cl_command_queue, nullptr );

            error = clReleaseMemObject( opencl_buffer );
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
