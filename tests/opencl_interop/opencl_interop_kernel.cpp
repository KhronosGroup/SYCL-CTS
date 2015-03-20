/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_kernel

namespace opencl_interop_kernel__
{
using namespace sycl_cts;

/** check inter-op types
*/
template <typename T>
using globalPtrType = cl::sycl::global_ptr<T>::pointer_t;
template <typename T>
using constantPtrType = cl::sycl::constant_ptr<T>::pointer_t;
template <typename T>
using localPtrType = cl::sycl::local_ptr<T>::pointer_t;
template <typename T>
using privatePtrType = cl::sycl::private_ptr<T>::pointer_t;
template <typename T>
using globalMultiPtrType = cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>::pointer_t;
template <typename T>
using constantMultiPtrType = cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>::pointer_t;
template <typename T>
using localMultiPtrType = cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>::pointer_t;
template <typename T>
using privateMultiPtrType = cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>::pointer_t;
template <typename T, int dims>
using vectorType = cl::sycl::vector<T, dims>::vector_t;

/** tests the kernel execution for OpenCL inter-op
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
            static const util::STRING kernelSource =
            R"(__kernel void test_kernel(__global int *argOne, read_only image2d_t arg2, sampler_t arg3, float arg4)
            {
                ;
            })";

            const size_t bufferSize = 32;
            int bufferData[bufferSize] = { 0 };

            const size_t imageSize = 1024;
            float imageData[imageSize] = { 0.0f };

            cl::sycl::queue queue;

            cl::sycl::buffer<int, bufferSize> buffer( bufferData, cl::sycl::range<1>(bufferSize) );

            cl::sycl::image<imageSize> image( imageData, cl::sycl::image_format::channel_order::RGBA, cl::sycl::image_format::channel_type::FLOAT, cl::sycl::range<2>(32, 32) );

            cl_program clProgram = nullptr;
            if ( !create_program( kernelSource, clProgram, log ) )
            {
                FAIL( log, "create_program failed" );
            }

            cl_kernel clKernel = nullptr;
            if ( !create_kernel( clProgram, "test_kernel", clKernel, log ) )
            {
                FAIL( log, "create_kernel failed" );
            }

            cl::sycl::kernel kernel( clKernel );

            queue.submit( [&](cl::sycl::handler &handler)
            {
                auto bufferAccessor = buffer.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>( handler );
                auto imageAccessor = image.get_access<cl::sycl::float4, cl::sycl::access::mode::read> ( handler );

                cl::sycl::sampler sampler(false, cl::sycl::sampler_addressing_mode::none, cl::sycl::sampler_filter_mode::nearest);
            
                /** check the set_arg() methods
                */
                handler.set_arg(0, bufferAccessor);
                handler.set_arg(1, imageAccessor);
                handler.set_arg(2, sampler);
                handler.set_arg(3, 15.0f);

                handler.single_task( kernel );
            } );
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

} /* namespace opencl_interop_kernel__ */
