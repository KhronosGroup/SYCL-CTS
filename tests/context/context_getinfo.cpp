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
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
    *  @param info, test_base::info structure as output
    */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
    *  @param log, test transcript logging class
    */
    virtual void run( util::logger &log )
    {
        try
        {
            cts_selector selector;
            cl::sycl::context context( selector );

            cl_uint info_uint;
            cl_bool info_bool;

            VECTOR_CLASS<cl_device_id> ctext_devices;
            VECTOR_CLASS<cl_context_properties> ctext_properties;

            info_uint = context.get_info<CL_CONTEXT_REFERENCE_COUNT>();
            info_uint = context.get_info<CL_CONTEXT_NUM_DEVICES>();
            ctext_devices = context.get_info<CL_CONTEXT_DEVICES>();
            ctext_properties = context.get_info<CL_CONTEXT_PROPERTIES>();
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

};  // sycl_cts
