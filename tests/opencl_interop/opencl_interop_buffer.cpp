/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME       opencl_interop_buffer

namespace sycl_cts
{
/** Test for the SYCL buffer OpenCL interoperation
    */
class TEST_NAME
    : public sycl_cts::util::test_base_opencl
{
public:

    /** return information about this test
    *  @param info, test_base::info structure as output
    */
    virtual void get_info(test_base::info & out) const
    {
        set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
    }

    /** execute this test
    *  @param log, test transcript logging class
    */
    virtual void run(util::logger & log)
    {
        try
        {
            const size_t size = 32;
            int data[size] = { 0 };

            cl_int error = CL_SUCCESS;

            cl_mem opencl_buffer = clCreateBuffer(
                m_cl_context,
                CL_MEM_READ_WRITE |
                CL_MEM_COPY_HOST_PTR,
                size * sizeof(int),
                data,
                &error);
            check_cl_success(
                error,
                __FILE__,
                __LINE__,
                log);

#if ENABLE_FULL_TEST
            cl::sycl::buffer<int, size> buffer(
                opencl_buffer,
                m_cl_command_queue,
                nullptr);
#endif
            error = clReleaseMemObject(opencl_buffer);
            check_cl_success(
                error,
                __FILE__,
                __LINE__,
                log);
        }
        catch (cl::sycl::sycl_error e)
        {
            log_exception(log, e);
            FAIL(log, "");
        }
    }
};

// register this test with the test_collection
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
