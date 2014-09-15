/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include <CL/cl.h>

// include the log class
#include "test_base.h"

// conformance test suite namespace
namespace sycl_cts
{
namespace util
{

/** Common base class for OpenCL inter operation tests
 */
class test_base_opencl
    : public sycl_cts::util::test_base
{
public:

    /** ctor
     */
    test_base_opencl();

    /** virtual destructor
     */
    virtual ~test_base_opencl( )
    {
        clReleaseCommandQueue(m_cl_command_queue);
        clReleaseContext(m_cl_context);
        free(m_platforms);
        free(m_devices);
    }

    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info & out ) const = 0;

    /** called before this test is executed
     *  @param log for emitting test notes and results
     */
    virtual bool setup( logger & log );

    /** execute this test
     *  @param log for emitting test notes and results
     */
    virtual void run( logger & log ) = 0;

    /** called after this test has executed
     */
    virtual void cleanup( );

    /** return a valid opencl platform object
     */
    cl_platform_id get_cl_platform_id( )
    {
        return m_cl_platform_id;
    }

    /** return a valid opencl cl_device_id object
     */
    cl_device_id get_cl_device_id( )
    {
        return m_cl_device;
    }

    /** return a valid opencl cl_context object
     */
    cl_context get_cl_context( )
    {
        return m_cl_context;
    }

    /** return a valid opencl cl_command_queue object
     */
    cl_command_queue get_cl_command_queue( )
    {
        return m_cl_command_queue;
    }

protected:

    /* instances of open cl objects */
    cl_platform_id      m_cl_platform_id;
    cl_device_id        m_cl_device;
    cl_context          m_cl_context;
    cl_command_queue    m_cl_command_queue;

private:
    /* Arrays containing results of queries.
     * Because OpenCL is a nasty C API and we
     * can't use stl things here, m_platforms
     * and m_devices are malloc'd. They are
     * freed in the dtor.
     */
    cl_platform_id *    m_platforms;
    cl_device_id *      m_devices;

}; // class test_base

}; // namespace util
}; // namespace sycl_cts
