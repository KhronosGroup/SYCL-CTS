/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "test_base_opencl.h"
#include "../tests/common/macros.h"
#include "../util/opencl_helper.h"

// conformance test suite namespace
namespace sycl_cts
{
namespace util
{

/** constructor which explicitly sets the opencl objects
 *  to nullptrs
 */
test_base_opencl::test_base_opencl()
    : m_cl_platform_id( nullptr )
    , m_cl_device( nullptr )
    , m_cl_context( nullptr )
    , m_cl_command_queue( nullptr )
{
}

/** called before this test is executed
 *  @param log for emitting test notes and results
 */
bool test_base_opencl::setup( logger &log )
{
    /* get the OpenCLHelper object */
    using sycl_cts::util::opencl_helper;
    using sycl_cts::util::get;
    opencl_helper &openclHelper = get<opencl_helper>();

    UNIQUE_PTR<cl_platform_id> platforms;
    UNIQUE_PTR<cl_device_id> devices;

    cl_uint N = 0;
    cl_int error = CL_SUCCESS;

    error = clGetPlatformIDs( 0, nullptr, &N );
    if ( !CHECK_CL_SUCCESS( log, error ) )
        return false;

    if ( N < 1 )
    {
        FAIL( log, "Unable to retrieve list of platforms via clGetPlatformIDs()" );
        return false;
    }

    platforms.reset( new cl_platform_id[N] );
    if ( platforms.get() == nullptr )
    {
        FAIL( log, "Malloc failed to allocate memory for platform list" );
        return false;
    }

    error = clGetPlatformIDs( N, platforms.get(), nullptr );
    if ( !CHECK_CL_SUCCESS( log, error ) )
        return false;

    error = clGetDeviceIDs( platforms.get()[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &N );
    if ( !CHECK_CL_SUCCESS( log, error ) )
        return false;

    if ( N < 1 )
    {
        FAIL( log, "Unable to retrieve list of devices via clGetDeviceIDs()" );
        return false;
    }

    devices.reset( new cl_device_id[N] );
    if ( devices.get() == nullptr )
    {
        FAIL( log, "Malloc failed to allocate memory for device list" );
        return false;
    }

    error = clGetDeviceIDs( platforms.get()[0], CL_DEVICE_TYPE_ALL, N, devices.get(), nullptr );
    if ( !CHECK_CL_SUCCESS( log, error ) )
        return false;

    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms.get()[0], 0 };

    m_cl_context = clCreateContext( properties, N, devices.get(), nullptr, nullptr, &error );
    if ( !CHECK_CL_SUCCESS( log, error ) )
        return false;

    // No special queue properties wanted
    m_cl_command_queue = clCreateCommandQueue( m_cl_context, devices.get()[0], 0, &error );
    if ( !CHECK_CL_SUCCESS( log, error ) )
        return false;

    m_cl_platform_id = platforms.get()[0];
    m_cl_device = devices.get()[0];

    return true;
}

/** called after this test has executed
 *  @param log for emitting test notes and results
 */
void test_base_opencl::cleanup()
{
    clReleaseCommandQueue( m_cl_command_queue );
    clReleaseContext( m_cl_context );
}

};  // namespace util
};  // namespace sycl_cts
