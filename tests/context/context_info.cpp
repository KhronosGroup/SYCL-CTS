/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME context_info

namespace contect_info__
{
using namespace sycl_cts;

/** tests the info for cl::sycl::context
 */
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
    */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
    */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cts_selector selector;
            cl::sycl::context context( selector );

            /** check types
            */
            using contextInfo = cl::sycl::info::context;

            /** initialize return values
            */
            cl_uint info_uint;
            cl_bool info_bool;

            /** check context info parameters
            */
            info_uint = context.get_info<cl::sycl::info::context::reference_count>();
            info_bool = context.get_info<cl::sycl::info::context::d3d10_prefer_shared_resources_khr>();
            info_bool = context.get_info<cl::sycl::info::context::d3d11_prefer_shared_resources_khr>();
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

} /* namespace context_info__ */
