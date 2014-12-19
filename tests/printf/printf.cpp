/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME printf

namespace printf__
{
using namespace sycl_cts;
using namespace cl::sycl;

#define MEANING 42.0f
#define SUCCESS 0

class functor_printf
{
    accessor<int, 1, access::read_write, access::global_buffer> a_dev;
public:
    functor_printf( accessor<int, 1, access::read_write, access::global_buffer> a )
        : a_dev(a)
    {}
    void operator() () {
        a_dev[0] = ::printf("Testing: %f \n", MEANING);
    }
};

void printf_on(device_selector& sel,int &i)
{
    queue queue(sel);
    buffer<int,1> a(&i, 1);
    command_group(queue, [&] () {
        auto a_dev = a.get_access<access::read_write>();
        functor_printf myKernel(a_dev);

        single_task(myKernel);
    });
}

/** test cl::sycl::kernel from functor
 */
class TEST_NAME : public util::test_base
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
            //host
            {
                int f = SUCCESS; //success
                cl::sycl::host_selector sel;
                printf_on(sel, f);
                if(f < SUCCESS)
                {
                     FAIL( log, "printf returned less than 0 on host_selector." );
                }
            }

            //default
            {
                int f = SUCCESS; //success
                cl::sycl::default_selector sel;
                printf_on(sel, f);
                if(f < SUCCESS)
                {
                     FAIL( log, "printf returned less than 0 on default_selector." );
                }
            }

            //gpu
            {
                int f = SUCCESS; //success
                cl::sycl::gpu_selector sel;
                printf_on(sel, f);
                if(f < SUCCESS )
                {
                     FAIL( log, "printf returned less than 0 on gpu_selector." );
                }
            }

            //cpu
            {
                int f = SUCCESS; //success
                cl::sycl::cpu_selector sel;
                printf_on(sel, f);
                if(f < SUCCESS)
                {
                     FAIL( log, "printf returned less than 0 on cpu_selector." );
                }
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

} /* namespace printf__ */
