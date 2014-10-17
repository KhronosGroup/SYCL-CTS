/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME context_getinfo

// conformance test suite namespace
namespace sycl_cts
{

    /**
    */
    class TEST_NAME
        : public util::test_base
    {
    public:

        /** return information about this test
        *  @param info, test_base::info structure as output
        */
        virtual void get_info(test_base::info & out) const
        {
            set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
        }

        /** execute the test
        *  @param log, test transcript logging class
        */
        virtual void run(util::logger & log)
        {
            try
            {
                cl::sycl::context context;
                cl_uint info_uint;
                cl_bool info_bool;

                info_uint = context.get_info<CL_CONTEXT_REFERENCE_COUNT>();
                info_uint = context.get_info<CL_CONTEXT_NUM_DEVICES>();
                /* 
                 * the next two are tricky - how does sycl return them?
                info_device_id * CL_CONTEXT_DEVICES
                info_context_properties * CL_CONTEXT_PROPERTIES
                 * These two macros don't exist in the header
                info_bool = context.get_info
                    <CL_CONTEXT_D3D10_PREFER_SHARED_RESOURCES_KHR>();
                info_bool = context.get_info
                    <CL_CONTEXT_D3D11_PREFER_SHARED_RESOURCES_KHR>();
                 */
            }
            catch (cl::sycl::sycl_error e)
            {
                log_exception(log, e);
                FAIL(log, "");
            }
        }

    };

    // construction of this proxy will register the above test
    static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
