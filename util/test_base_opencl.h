/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include <CL/cl.h>

#include "stl.h"
#include "test_base.h"

// conformance test suite namespace
namespace sycl_cts
{
namespace util
{

/** Common base class for OpenCL inter operation tests
 */
class test_base_opencl : public sycl_cts::util::test_base
{
public:
    /** ctor
     */
    test_base_opencl();

    /** virtual destructor
     */
    virtual ~test_base_opencl()
    {
    }

protected:

    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info &out ) const = 0;

    /** called before this test is executed
     *  @param log for emitting test notes and results
     */
    virtual bool setup( logger &log );

    /** execute this test
     *  @param log for emitting test notes and results
     */
    virtual void run( logger &log ) = 0;

    /** called after this test has executed
     */
    virtual void cleanup();

    /** return a valid opencl platform object
     */
    cl_platform_id get_cl_platform_id()
    {
        return m_cl_platform_id;
    }

    /** return a valid opencl cl_device_id object
     */
    cl_device_id get_cl_device_id()
    {
        return m_cl_device;
    }

    /** return a valid opencl cl_context object
     */
    cl_context get_cl_context()
    {
        return m_cl_context;
    }

    /** return a valid opencl cl_command_queue object
     */
    cl_command_queue get_cl_command_queue()
    {
        return m_cl_command_queue;
    }

    /** create an opencl program
     */
    bool create_program
    (
        const STRING & source     ,
        cl_program   & outProgram,
        logger       & log
    );

    /** create and opencl kernel
     */
    bool create_kernel
    (
        const cl_program & clProgram,
        const STRING     & name     ,
        cl_kernel        & outKernel,
        logger           & log
    );


    /* instances of open cl objects */
    cl_platform_id   m_cl_platform_id;
    cl_device_id     m_cl_device;
    cl_context       m_cl_context;
    cl_command_queue m_cl_command_queue;
    cl_sampler       m_cl_sampler;

    /*  */
    VECTOR<cl_kernel>  m_openKernels;
    VECTOR<cl_program> m_openPrograms;

};  // class test_base

}  // namespace util
}  // namespace sycl_cts
