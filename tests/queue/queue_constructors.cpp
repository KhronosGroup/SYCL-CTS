/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME queue_constructors

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
    virtual void get_info( test_base::info & out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger & log )
    {
        try
        {
            cl::sycl::queue queue;

            // construct the cts default selector
            cts_selector selector;
            cl::sycl::queue queue_ds(selector);

#if ENABLE_FULL_TEST
            cl::sycl::device device(selector);
            cl::sycl::queue queue_device(device);
#endif

            cl::sycl::context context;
            cl::sycl::queue queue_ctext_ds(context, selector);

#if ENABLE_FULL_TEST
            cl::sycl::queue queue_copy(queue);
#endif
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "" );
        }
    }

};

// register this test with the test_collection
static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
