/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_info

namespace queue_info__
{
using namespace sycl_cts;

/** tests the info for cl::sycl::queue
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

    /** execute this test
    */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cts_selector selector;
            cl::sycl::queue queue( selector );

            /** check types
            */
            using queueInfo = cl::sycl::info::queue;

            /** initialize return values
            */
            cl_uint refCount;

            /** check device info parameters
            */
            refCount = queue.get_info<cl::sycl::info::queue::reference_count>();
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

} /* namespace queue_info__ */
