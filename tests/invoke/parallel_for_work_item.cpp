/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME parallel_for_work_item

namespace parallel_for_work_item__
{
using namespace sycl_cts;

/** test cl::sycl::parallel_for_work_item
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
        static const int SIZE = 16;

        try
        {
            int f[SIZE * SIZE * SIZE];
            {
                using namespace cl::sycl;
                default_selector sel;
                queue queue(sel);

                buffer<int, 1> buf( f, range<1>(SIZE * SIZE * SIZE) );
                queue.submit( [&] ( handler& cgh )
                {
                    auto a_dev = buf.get_access<cl::sycl::access::mode::read_write>( cgh );

                    cgh.parallel_for_workgroup<class TEST_NAME>( nd_range<3>( range<3>(SIZE, SIZE, SIZE),
                                                                          range<3>(1, 1, 1) ),
                                                             [=] (group<3> myGroup)
                    {
                        cgh.parallel_for_workitem( myGroup, [=] (item<3> item )
                        {
                            //the local sizes should be
                            a_dev[item.get_global().get_global_linear()] = item.get_global().get_global_linear();
                        } );
                    } );
                } );

                queue.wait_and_throw();

            }

            for ( int i = 0; i < SIZE * SIZE * SIZE; i++ )
                if ( !CHECK_VALUE( log, f[i], SIZE, i ) )
                    return;

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

} /* namespace kernel_as_functor__ */
