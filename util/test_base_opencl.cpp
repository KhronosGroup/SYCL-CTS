/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "test_base_opencl.h"
#include "../tests/common/macros.h"

// conformance test suite namespace
namespace sycl_cts
{
namespace util
{

/** constructor which explicitly sets the opencl objects
 *  to nullptrs
 */
test_base_opencl::test_base_opencl() :
    m_cl_platform_id(nullptr),
    m_cl_device(nullptr),
    m_cl_context(nullptr),
    m_cl_command_queue(nullptr),
    m_platforms(nullptr),
    m_devices(nullptr)
{
    ;
}

void test_base_opencl::check_cl_success(
    cl_int error,
    const char * file,
    int line,
    logger & log)
{
    if (error != CL_SUCCESS)
    {
        std::string err_msg("Expected CL_SUCCESS in ");
        err_msg.append(file);
        err_msg.append(":");
        err_msg.append(std::to_string(line));
        err_msg.append(", got ");
        err_msg.append(std::to_string(error));
        FAIL( log, err_msg );
    }
}

/** called before this test is executed
 *  @param log for emitting test notes and results
 */
bool test_base_opencl::setup( logger & log )
{
    cl_uint N = 0;
    cl_int error = CL_SUCCESS;

    error = clGetPlatformIDs(
        0,
        nullptr,
        &N);
    check_cl_success(error,__FILE__,__LINE__,log);

    if (N < 1)
    {
        FAIL( log, "Unable to retrieve list of platforms via clGetPlatformIDs()" );
        return false;
    }

    m_platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * N);
    if(m_platforms == nullptr)
    {
        FAIL( log, "Malloc failed to allocate memory for platform list" );
        return false;
    }

    error = clGetPlatformIDs(
        N,
        m_platforms,
        nullptr);
    check_cl_success(error, __FILE__, __LINE__, log);

    error = clGetDeviceIDs(
        m_platforms[0],
        CL_DEVICE_TYPE_ALL,
        0,
        nullptr,
        &N);
    check_cl_success(error, __FILE__, __LINE__, log);

    if(N < 1)
    {
        FAIL( log, "Unable to retrieve list of devices via clGetDeviceIDs()" );
        return false;
    }

    m_devices = (cl_device_id *) malloc(sizeof(cl_device_id) * N);
    if(m_devices == nullptr)
    {
        FAIL( log, "Malloc failed to allocate memory for device list" );
        return false;
    }

    error = clGetDeviceIDs(
        m_platforms[0],
        CL_DEVICE_TYPE_ALL,
        N,
        m_devices,
        nullptr);
    check_cl_success(error, __FILE__, __LINE__, log);

    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) m_platforms[0],
        0 };
    m_cl_context = clCreateContext(
        properties,
        N,
        m_devices,
        nullptr,
        nullptr,
        &error);
    check_cl_success(error, __FILE__, __LINE__, log);

    // No special queue properties wanted
    m_cl_command_queue = clCreateCommandQueue(
        m_cl_context,
        m_devices[0],
        0,
        &error);
    check_cl_success(error, __FILE__, __LINE__, log);

    m_cl_platform_id    = m_platforms[0];
    m_cl_device         = m_devices[0];

    return true;
}

/** called after this test has executed
 *  @param log for emitting test notes and results
 */
void test_base_opencl::cleanup( )
{
    // stub
}

}; // namespace util
}; // namespace sycl_cts
