/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_using_program_object

namespace kernel_using_program_object__
{
using namespace sycl_cts;
using namespace cl::sycl;

#define MEANING 42.0f

void work_function(util::logger &log, buffer<float, 1> &a)
{
    class my_kernel;
    intel_selector sel;
    queue queue(sel);
    program my_program(queue.get_context());

    my_program.build_kernel_from_name<my_kernel>();

    command_group(queue, [&] () {
        auto a_dev = a.get_access<access::read_write>();

        parallel_for<class my_kernel>( nd_range<1>(1),
                                    my_program,
                                    ([=] () {
                                        a_dev[0] = MEANING;
                                    }));
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
                work_function(log, buf);
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

} /* namespace kernel_using_program_object__ */
