/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_constructors

namespace kernel_constructors__
{
using namespace sycl_cts;

/** simple test kernel
    */
util::STRING kernel_source = R"(
__kernel void sample(__global float * input)
{
    input[get_global_id(0)] = get_global_id(0);
}
)";

/** test cl::sycl::kernel
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl
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
            cl_program cl_program = nullptr;
            if ( !create_program( kernel_source, cl_program, log ) )
                return;

            cl_kernel cl_kernel = nullptr;
            if ( !create_kernel( cl_program, "sample", cl_kernel, log ) )
                return;

            cl::sycl::kernel k( cl_kernel );
            cl::sycl::kernel k_copy( k );
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

} /* namespace kernel_constructors__ */
