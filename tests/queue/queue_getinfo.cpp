/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_getinfo

namespace queue_getinfo__
{
using namespace sycl_cts;

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

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {
        try
        {
            // construct the cts default selector
            cts_selector selector;

            cl::sycl::queue queue( selector );

            cl_context ctext = queue.get_info<CL_QUEUE_CONTEXT>();
            cl_device_id dev = queue.get_info<CL_QUEUE_DEVICE>();
            cl_uint refcount = queue.get_info<CL_QUEUE_REFERENCE_COUNT>();
            cl_command_queue_properties properties = queue.get_info<CL_QUEUE_PROPERTIES>();
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

} /* namespace queue_getinfo__ */
