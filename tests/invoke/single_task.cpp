/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME single_task_invoke

namespace single_task__
{
using namespace sycl_cts;

#define MEANING 42.0f

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
            using namespace cl::sycl;
            default_selector sel;
            queue queue(sel);

            float f = 0;
            {
                cl::sycl::buffer<float,1> buf(&f, cl::sycl::range<1>(1));
                command_group(queue, [&] () {
                    auto a_dev = buf.get_access<access::read_write>();

                    single_task<class TEST_NAME>( [=] () {
                        a_dev[0] = MEANING;
                    });
                });
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

} /* namespace single_task__ */
