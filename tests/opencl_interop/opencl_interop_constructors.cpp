/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_constructors

namespace opencl_interop_constructors__
{
using namespace sycl_cts;

/** tests all of the constructors for OpenCL inter-op
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
        try
        {
            cl::sycl::string_class kernelSource = R"(
            __kernel void sample(__global float *input)
            {
              input[get_global_id(0)] = get_global_id(0);
            }
            )";

            /** check platform (platform_id) constructor
            */
            {
                cl::sycl::platform platform( m_cl_platform_id );

                cl::sycl::info::platform_id platformID = platform.get()
                if ( platformID != m_cl_platform_id )
                {
                    FAIL( log, "platform was not constructed correctly" );
                }
            }

            /** check device (device_id) constructor
            */
            {
                cl::sycl::device device( m_cl_device_id );

                cl::sycl::info::device_id deviceID = device.get()
                if ( deviceID != m_cl_device_id )
                {
                    FAIL( log, "device was not constructed correctly" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseDevice( deviceID ) ) )
                {
                    FAIL( log, "failed to release OpenCL device" );
                }
            }

            /** check context (context_id) constructor
            */
            {
                cl::sycl::context context( m_cl_context );

                cl::sycl::info::context_id contextID = context.get()
                if ( contextID != m_cl_context )
                {
                    FAIL( log, "context was not constructed correctly" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseContext( contextID ) ) )
                {
                    FAIL( log, "failed to release OpenCL context" );
                }
            }

            /** check queue (queue_id) constructor
            */
            {
                cl::sycl::queue queue( m_cl_command_queue );

                cl::sycl::info::queue_id queueID = queue.get()
                if ( queueID != m_cl_command_queue )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseCommandQueue( queueID ) ) )
                {
                    FAIL( log, "failed to release OpenCL command queue" );
                }
            }

            /** check program (context, program_id) constructor
            */
            {
                cl::sycl::info::program_id clProgram = nullptr;
                if ( !create_program( kernelSource, clProgram, log ) )
                {
                    FAIL( log, "create_program failed" );
                }

                cts_selector selector;
                cl::sycl::context context( selector );
                cl::sycl::program program(context, clProgram);

                cl::sycl::info::program_id programID = program.get()
                if ( programID != clProgram )
                {
                    FAIL( log, "program was not constructed correctly" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseProgram( programID ) ) )
                {
                    FAIL( log, "failed to release OpenCL program" );
                }
            }

            /** check kernel (kernel_id) constructor
            */
            {
                cl::sycl::info::program_id clProgram = nullptr;
                if ( !create_program( kernelSource, clProgram, log ) )
                {
                    FAIL( log, "create_program failed" );
                }

                cl::sycl::info::kernel_id clKernel = nullptr;
                if ( !create_kernel( clProgram, "sample", clKernel, log ) )
                {
                    FAIL( log, "create_kernel failed" );
                }

                cl::sycl::kernel kernel( clKernel );

                cl::sycl::info::kernel_id kernelID = kernel.get()
                if ( kernelID != clProgram )
                {
                    FAIL( log, "kernel was not constructed correctly" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseKernel( kernelID ) ) )
                {
                    FAIL( log, "failed to release OpenCL kernel" );
                }
            }

            

            

            /** check sampler (sampler_id) constructor
            */
            {
              cl::sycl::sampler sampler( m_cl_sampler );
              cl::sycl::info::sampler_id clSampler = sampler.get()
              if ( clSampler != m_cl_sampler )
              {
                  FAIL( log, "sampler was not constructed correctly" );
              }
              if ( !CHECK_CL_SUCCESS( log, clReleaseSampler( clSampler ) ) )
              {
                  FAIL( log, "failed to release OpenCL sampler" );
              }
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "a sycl exception was caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_constructors__ */
