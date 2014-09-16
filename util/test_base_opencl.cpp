/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "test_base_opencl.h"

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

// TODO: move this somewhere useful!
void checkCLSuccess(cl_int error)
{
    // TODO: do something meaningful here!
    if(error != CL_SUCCESS)
        ;
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
    checkCLSuccess(error);

    // TODO: deal with this case!
    if(N < 1)
        ;

    m_platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * N);
    if(m_platforms == nullptr)
    {
        log.fail("Malloc failed to allocate memory for platform list");
        return false;
    }

    error = clGetPlatformIDs(
        N,
        m_platforms,
        nullptr);
    checkCLSuccess(error);

    error = clGetDeviceIDs(
        m_platforms[0],
        CL_DEVICE_TYPE_ALL,
        0,
        nullptr,
        &N);
    checkCLSuccess(error);

    // TODO: deal with this case!
    if(N < 1)
        ;

    m_devices = (cl_device_id *) malloc(sizeof(cl_device_id) * N);
    if(m_devices == nullptr)
    {
        log.fail("Malloc failed to allocate memory for device list");
        return false;
    }

    error = clGetDeviceIDs(
        m_platforms[0],
        CL_DEVICE_TYPE_ALL,
        N,
        m_devices,
        nullptr);
    checkCLSuccess(error);

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
    checkCLSuccess(error);

    // No special queue properties wanted
    m_cl_command_queue = clCreateCommandQueue(
        m_cl_context,
        m_devices[0],
        0,
        &error);
    checkCLSuccess(error);

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
