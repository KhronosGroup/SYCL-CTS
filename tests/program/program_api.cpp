/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_api

namespace program_api__
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
            using namespace cl::sycl;

            context context;
            {
                program prog(context);
                if(prog.get_devices().size() < 1)
                {
                    FAIL( log, "Wrong value for program.get_devices()" );
                }
            }

            //build program without compile_options
            {
                class build_without_options;
                program prog(context);

                int in = 1;
                int out = 0;

                prog.build_kernel_from_name<build_without_options>();

                //check for get_binaries()
                if(prog.get_binaries().size() < 1)
                {
                    FAIL( log, "Wrong value for program.get_binaries()" );
                }

                for(int i =0; i < 42; i++)
                {
                    {
                        command_group(queue, [&] () {
                            auto in_dev = in.get_access<access::read>();
                            auto out_dev = out.get_access<access::write>();

                            parallel_for<class build_without_options>( nd_range<1>(1), prog,
                            ([=] () {
                                out_dev[0] = out_dev[0] + in_dev[0];
                            }));
                        });
                    }
                }
            }

            //build program with compile_options
            {
                class build_with_options;
                program prog(context);

                int in = 1;
                int out = 0;

                prog.build_kernel_from_name<build_with_options>("-cl-opt-disable");

                //check for get_binaries()
                if(prog.get_binaries().size() < 1)
                {
                    FAIL( log, "Wrong value for program.get_binaries()" );
                }

                //check for get_build_options()
                if(prog.get_build_options().length() == 0)
                {
                    FAIL( log, "program.get_build_options() shouldn't be empty" );
                }

                for(int i =0; i < 42; i++)
                {
                    {
                        command_group(queue, [&] () {
                            auto in_dev = in.get_access<access::read>();
                            auto out_dev = out.get_access<access::write>();

                            parallel_for<class build_with_options>( nd_range<1>(1), prog,
                            ([=] () {
                                out_dev[0] = out_dev[0] + in_dev[0];
                            }));
                        });
                    }
                }

                {
                    class compile_without_options; //Forward declaration of the name of the lambda functor
                    cl::sycl::queue myQueue;
                    // obtain an existing OpenCL C program object

                    cl_program myClProgram = nullptr;
                    if ( !create_program( kernel_source, myClProgram, log ) )
                        FAIL( log, "Didn't create the cl_program" );

                    // Create a SYCL program object from a cl_program object
                    cl::sycl::program myExternProgram(myQueue.get_context(), myClProgram);

                    // Add in the SYCL program object for our kernel
                    cl::sycl::program mySyclProgram (myQueue.get_context ());
                    mySyclProgram.compile_from_kernel_name<compile_without_options>("-cl-opt-disable");

                    // Link myClProgram with the SYCL program object
                    cl::sycl::program myLinkedProgram ({myExternProgram,
                    mySyclProgram}, "-cl-fast-relaxed-mat");

                    cl::sycl::command_group(myQueue, [&] () {
                        cl::sycl::parallel_for<class compile_without_options>(cl::sycl::nd_range<2>(4,4),
                            myLinkedProgram, // execute the kernel as compiled in MyProgram
                            ([=] (cl::sycl::item index) {
                                //[kernel code]
                            }));
                    });
                }

                {
                    class compile_without_options; //Forward declaration of the name of the lambda functor
                    cl::sycl::queue myQueue;
                    // obtain an existing OpenCL C program object

                    cl_program myClProgram = nullptr;
                    if ( !create_program( kernel_source, myClProgram, log ) )
                        FAIL( log, "Didn't create the cl_program" );

                    // Create a SYCL program object from a cl_program object
                    cl::sycl::program myExternProgram(myQueue.get_context(), myClProgram);

                    // Add in the SYCL program object for our kernel
                    cl::sycl::program mySyclProgram (myQueue.get_context ());
                    mySyclProgram.compile_from_kernel_name<compile_without_options>();

                    // Link myClProgram with the SYCL program object
                    cl::sycl::program myLinkedProgram ({myExternProgram,
                    mySyclProgram});

                    cl::sycl::command_group(myQueue, [&] () {
                        cl::sycl::parallel_for<class compile_without_options>(cl::sycl::nd_range<2>(4,4),
                            myLinkedProgram, // execute the kernel as compiled in MyProgram
                            ([=] (cl::sycl::item index) {
                                //[kernel code]
                            }));
                    });
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

} /* namespace program_api__ */
