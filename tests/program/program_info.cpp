/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_info

namespace program_info__
{
using namespace sycl_cts;

/** tests the info for cl::sycl::program
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

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cts_selector selector;
            cl::sycl::context context;
            cl::sycl::program program( context );

            /** check types
            */
            using programID = cl::sycl::info::program_id;
            using programInfo = cl::sycl::info::program;

            /** initialize return values
            */
            cl_uint referenceCount;

            /** check program info parameters
            */
            referenceCount = program.get_info<cl::sycl::info::program::reference_count>();
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace program_info__ */
