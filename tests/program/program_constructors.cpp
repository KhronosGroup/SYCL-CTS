/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_constructors

namespace program_constructors__
{
using namespace sycl_cts;
using namespace cl::sycl;

/** simple test kernel
 */
util::STRING kernel_source = R"(
__kernel void sample(__global float * input)
{
    input[get_global_id(0)] = get_global_id(0);
}
)";

/** test cl::sycl::program
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
            context cxt;

            //context
            {
                program prog(cxt);
                UNUSED(prog);
            }

            //context, vector_class<device>
            {
                device d;
                auto devices = d.get_devices();
                program prog(cxt, devices);
                UNUSED(prog);
            }

            //context, cl_program
            {
                cl_program cl_program = nullptr;
                if ( !create_program( kernel_source, cl_program, log ) )
                    FAIL( log, "create_program failed." );
                program prog(ctx, cl_program);
                UNUSED(prog);
            }

            //vector_class<program>, string_class
            {
                std::vector<program> programs;

                programs.push_back(program(cxt));
                programs.push_back(program(cxt));

                program prog_no_linking_options(programs);
                program prog_with_linking_options(programs,
                                                 "-cl-fast-relaxed-mat");
                UNUSED(prog_no_linking_options);
                UNUSED(prog_with_linking_options);
            }

            //const program&
            {
                program prog_a(cxt);
                program prog_b(prog_a);
                UNUSED(prog_b);
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

} /* namespace program_constructors__ */
