/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_api

namespace program_api__
{
using namespace sycl_cts;
using namespace cl::sycl;

/** simple OpenCL test kernel
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
     */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
     */
    virtual void run( util::logger &log ) override
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
                queue myQueue;
                program prog(myQueue.get_context());

                int in = 1;
                int out = 0;

                prog.build_from_kernel_name<build_without_options>();

                //check for get_binaries()
                if(prog.get_binaries().size() < 1)
                {
                    FAIL( log, "Wrong value for program.get_binaries()" );
                }

                buffer<int, 1> bIn(&in, range<1>(1));
                buffer<int, 1> bOut(&out, range<1>(1));

                for(int i = 0; i < 10; i++)
                {
                    {
                        myQueue.submit( [&]( handler& cgh )
                        {
                            auto in_dev = bIn.get_access<cl::sycl::access::mode::read>( cgh );
                            auto out_dev = bOut.get_access<cl::sycl::access::mode::write>( cgh );

                            cgh.parallel_for<class build_without_options>(
                              prog.get_kernel<build_without_options>(), nd_range<1>(range<1>(1)), [=](cl::sycl::item<1> i)
                            {
                                out_dev[0] = out_dev[0] + in_dev[0];
                            } );
                        } );
                    }
                }
            }

            //build program with compile_options
            {
                class build_with_options;
                queue myQueue;
                program prog(myQueue.get_context());

                int in = 1;
                int out = 0;

                prog.build_from_kernel_name<build_with_options>("-cl-opt-disable");

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

                buffer<int, 1> bIn(&in, range<1>(1));
                buffer<int, 1> bOut(&out, range<1>(1));

                for(int i =0; i < 10; i++)
                {
                    {
                        myQueue.submit( [&]( handler& cgh )
                        {
                            auto in_dev = bIn.get_access<cl::sycl::access::mode::read>( cgh );
                            auto out_dev = bOut.get_access<cl::sycl::access::mode::write>( cgh );

                            cgh.parallel_for<class build_with_options>(
                               prog.get_kernel<build_with_options>(), nd_range<1>(range<1>(1)), [=](cl::sycl::item<1> i)
                            {
                                out_dev[0] = out_dev[0] + in_dev[0];
                            } );
                        } );
                    }
                }

                
                /// taken from page 98 section 3.7.2.3
                {
                    class compile_with_options; //Forward declaration of the name of the lambda functor
                    cl::sycl::queue myQueue;
                    // obtain an existing OpenCL C program object

                    cl_program myClProgram = nullptr;
                    if ( !create_program( kernel_source, myClProgram, log ) )
                        FAIL( log, "Didn't create the cl_program" );

                    // Create a SYCL program object from a cl_program object
                    cl::sycl::program myExternProgram(myQueue.get_context(), myClProgram);

                    // Add in the SYCL program object for our kernel
                    cl::sycl::program mySyclProgram (myQueue.get_context ());
                    mySyclProgram.compile_from_kernel_name<compile_with_options>("-cl-opt-disable");

                    // Link myClProgram with the SYCL program object
                    cl::sycl::program myLinkedProgram ({myExternProgram,
                    mySyclProgram}, "-cl-fast-relaxed-mat");

                    myQueue.submit( [&] (cl::sycl::handler& handler) {
                        cl::sycl::parallel_for<class compile_with_options>(
                          myLinkedProgram.get_kernel<compile_with_options>(),
                          cl::sycl::nd_range<2>(range<2>(4,4)), // execute the kernel as compiled in MyProgram
                            ([=] (cl::sycl::item<2> index) {
                                //[kernel code]
                            }));
                    });

                    myQueue.wait_and_throw();

                }

                
                /// taken from page 98 section 3.7.2.3
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

                    myQueue.submit( [&] (cl::sycl::handler& handler) {
                        cl::sycl::parallel_for<class compile_without_options>(
                          myLinkedProgram.get_kernel<compile_without_options>(),
                          cl::sycl::nd_range<2>(range<2>(4,4)), // execute the kernel as compiled in MyProgram
                            ([=] (cl::sycl::item<2> index) {
                                //[kernel code]
                            }));
                    });

                    myQueue.wait_and_throw();

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
