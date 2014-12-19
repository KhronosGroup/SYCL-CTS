/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_from_functor

namespace kernel_as_functor__
{
using namespace sycl_cts;
using namespace cl::sycl;

#define MEANING 42.0f

class functor
{
    accessor<float, 1, access::read_write, access::global_buffer> a_dev;
public:
    functor( accessor<float, 1, access::read_write, access::global_buffer> a )
        : a_dev(a)
    {}
    void operator() () {
        a_dev[0] = MEANING;
    }
};

void work_function(buffer<float, 1> &a)
{
    intel_selector sel;
    queue queue(sel);

    command_group(queue, [&] () {
        auto a_dev = a.get_access<access::read_write>();
        functor kernel(a_dev);

        single_task(kernel);
    });
}

/** test cl::sycl::kernel from functor
 */
class TEST_NAME : public sycl_cts::util::test_base
{
public:

    /** return information about this test
     *  @param out, test_base::info structure as output
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
            float f = 0;
            {
                cl::sycl::buffer<float,1> buf(&f, cl::sycl::range<1>(1));
                work_function(buf);
            }

            if(f != MEANING)
            {
                FAIL( log, "buffer returned wrong value." );
            }

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
