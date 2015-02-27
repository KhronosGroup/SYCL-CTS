/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME single_task_invoke

namespace single_task__
{
using namespace sycl_cts;

/** test cl::sycl::kernel from functor
 */
class TEST_NAME : public sycl_cts::util::test_base
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
        static const int MEANING = 42;

        try
        {
            using namespace cl::sycl;
            default_selector sel;
            queue queue(sel);

            int f = 0;
            {
                cl::sycl::buffer<int, 1> buf( &f, cl::sycl::range<1>( 1 ) );
                queue.submit( [&]( handler& cgh )
                {
                    auto a_dev = buf.get_access<cl::sycl::access::mode::read_write>( cgh );

                    cgh.single_task<class TEST_NAME>( [=]()
                    {
                        a_dev[0] = MEANING;
                    } );
                } );
            }

            if(f != MEANING)
                FAIL( log, "buffer returned wrong value." );

            queue.wait_and_throw();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace single_task__ */
