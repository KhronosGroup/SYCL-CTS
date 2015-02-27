/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_get

namespace opencl_interop_get__
{
using namespace sycl_cts;

/** tests all of the get() methods for OpenCL inter-op
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
            cts_selector selector;

            /** check platform get() method
            */
            cl::sycl::platform platform( selector );
            auto platformID = platform.get();
            if ( typeid( platformID ) != typeid( cl::sycl::info::platform_id ) )
            {
                FAIL( log, "get() does not return cl::sycl::info::platform_id." );
            }
            if ( platform.is_host() )
            {
                if ( platformID != 0 )
                {
                    FAIL( log, "platform is in host mode but get() did not return a nullptr" );
                }
            }
            else
            {
                if ( platformID == 0 )
                {
                    FAIL( log, "get() did not return a valid platform_id" );
                }
            }

            /** check device get() method
            */
            cl::sycl::device device( selector );
            auto deviceID = device.get();
            if ( typeid( deviceID ) != typeid( cl::sycl::info::device_id ) )
            {
                FAIL( log, "get() does not return device_id" );
            }
            if ( device.is_host() )
            {
                if ( deviceID != 0 )
                {
                    FAIL( log, "device is in host mode but get() did not return a nullptr" );
                }
            }
            else
            {
                if ( deviceID == 0 )
                {
                    FAIL( log, "get() did not return a valid device_id" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseDevice( deviceID ) ) )
                {
                    FAIL( log, "failed to release the device_id" );
                }
            }

            /** check context get() method
            */
            cl::sycl::context context( selector );
            auto contextID = context.get();
            if ( typeid( contextID ) != typeid( cl::sycl::info::context_id ) )
            {
                FAIL( log, "get() does not return context_id" );
            }
            if ( context.is_host() )
            {
                if ( contextID != nullptr )
                {
                    FAIL( log, "context is in host mode but get() did not return a nullptr" );
                }
            }
            else
            {
                if ( contextID == nullptr )
                {
                    FAIL( log, "get() did not return a valid context_id" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseContext( contextID ) ) )
                {
                    FAIL( log, "failed to release the context_id" );
                }
            }

            /** check queue get() method
            */
            cl::sycl::queue( selector );
            auto queueID = queue.get();
            if ( typeid( queueID ) != typeid( cl::sycl::info::context_id ) )
            {
                FAIL( log, "get() does not return queue_id" );
            }
            if ( queue.is_host() )
            {
                if ( queueID != nullptr )
                {
                    FAIL( log, "queue is in host mode but get() did not return a nullptr" );
                }
            }
            else
            {
                if ( queueID == nullptr )
                {
                    FAIL( log, "get() did not return a valid queue_id" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseCommandQueue( queueID ) ) )
                {
                    FAIL( log, "failed to release the queue_id" );
                }
            }

            /** check buffer accessor get() method
            */
            cl::sycl::buffer<int, 1> buffer( cl::sycl::range<1>( 1 ) );
            queue.submit(
            [&]( cl::sycl::handler &handler )
            {
                cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> accessor(
                    buffer, handler );

                auto bufferID = accessor.get();
                if ( typeid( bufferID ) != typeid( cl::sycl::info::buffer_id ) )
                {
                    FAIL( log, "get() does not return buffer_id" );
                }
                if ( bufferID == nullptr )
                {
                    FAIL( log, "get() returned an invalid buffer_id" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseMemObject( bufferID ) ) )
                {
                    FAIL( log, "failed to release the buffer_id" );
                }
            } );

            /** check image accessor get() method
            */
            char data[256];
            cl::sycl::image<2> image( data, cl::sycl::image_format::channel_order::rgba,
                                      cl::sycl::image_format::channel_type::unsigned_int_8, cl::sycl::range<2>( 8, 8 ) );
            queue.submit( [&]( cl::sycl::handler &handler )
            {
                cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::image> accessor(
                    image, handler );

                auto imageID = accessor.get();
                if ( typeid( imageID ) != typeid( cl::sycl::info::image_id ) )
                {
                    FAIL( log, "get() does not return image_id" );
                }
                if ( imageID == nullptr )
                {
                    FAIL( log, "get() returned an invalid image_id" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseMemObject( imageID ) ) )
                {
                    FAIL( log, "failed to release the image_id" );
                }
            } );

            /** check sampler get() method
            */
            queue.submit( [&]( cl::sycl::handler &handler )
            {
                cl::sycl::sampler sampler(false, cl::sycl::sampler_addressing_mode::clamp_to_edge, cl::sycl::sampler_filter_mode::nearest)

                auto samplerID = sampler.get();
                if ( typeid( samplerID ) != typeid( cl::sycl::info::sampler_id ) )
                {
                    FAIL( log, "get() does not return sampler_id" );
                }
                if ( samplerID == nullptr )
                {
                    FAIL( log, "get() returned an invalid sampler_id" );
                }
                if ( !CHECK_CL_SUCCESS( log, clReleaseSampler( samplerID ) ) )
                {
                    FAIL( log, "failed to release teh sampler_id" );
                }
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

} /* namespace opencl_interop_get__ */
