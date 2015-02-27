/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_as_functor

namespace kernel_as_functor__
{
using namespace sycl_cts;
using namespace cl::sycl;

template <int32_t test_value>
class functor
{
    accessor<int32_t, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> a_dev;

public:
    functor( accessor<int32_t, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> a )
        : a_dev(a)
    {
    }

    void operator()()
    {
        a_dev[0] = test_value;
    }
};

template <int32_t test_value>
void work_function( buffer<int32_t, 1> &a )
{
    cts_selector sel;
    queue queue( sel );

    queue.submit( [&]( handler& cgh )
    {
        auto a_dev = a.get_access<cl::sycl::access::mode::read_write>( cgh );
        functor<test_value> kernel( a_dev );

        cgh.single_task( kernel );
    } );

    queue.wait_and_throw();

}

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
        static const int32_t test_value = 42;

        try
        {
            int32_t memory = 0;
            {
                cl::sycl::buffer<int32_t, 1> buf( &memory, cl::sycl::range<1>( 1 ) );
                work_function<test_value>(buf);
            }

            if ( memory != test_value)
                FAIL( log, "buffer returned wrong value." );

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
