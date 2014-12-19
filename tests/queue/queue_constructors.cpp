/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_constructors

namespace queue_constructor__
{
using namespace sycl_cts;

template <typename T> using function_class = std::function<T>;

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
            function_class<void(cl::sycl::exception_list)> fn =
            [&](cl::sycl::exception_list l)
            {
                if (l.size() > 1)
                    FAIL(log, "Exception thrown during execution of kernel");
            };

            cl::sycl::queue queue;
            cl::sycl::queue queue_handler(fn);

            // construct the cts default selector
            cts_selector selector;
            cl::sycl::queue queue_ds( selector );
            cl::sycl::queue queue_ds_handler( selector, fn );
            cl::sycl::device device( selector );
            cl::sycl::queue queue_device( device );
            cl::sycl::queue queue_device_handler( device, fn );

            cl::sycl::context context;
            cl::sycl::queue queue_ctext_ds( context, selector );
            cl::sycl::queue queue_ctext_ds_handler( context, selector, fn );
            cl::sycl::queue queue_ctext_dev( context, device );
            cl::sycl::queue queue_ctext_dev_handler( context, device, fn);
            cl::sycl::queue queue_copy( queue );
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

} /* namespace queue_constructors */
