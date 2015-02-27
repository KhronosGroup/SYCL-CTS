/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_info

namespace kernel_info__
{
using namespace sycl_cts;

class kernel0;

/** tests the info for cl::sycl::kernel
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
            cts_selector selector;
            cl::sycl::queue queue( selector );
            cl::sycl::program program( queue.get_context() );
            program.build_from_kernel_name<kernel0>();
            cl::sycl::kernel kernel = program.get_kernel<kernel0>();
            queue.submit( [&]( handler &cgh )
            {
                cgh.single_task<kernel0>( [=]()
                {
                } );
            } );

            /** check types
            */
            using kernelID = cl::sycl::info::kernel_id;
            using kernelInfo = cl::sycl::info::kernel;

            /** initialize return values
            */
            cl_uint clUintRet;

            /** check program info parameters
            */
            clUintRet = kernel.get_info<cl::sycl::info::kernel::reference_count>();
            clUintRet = kernel.get_info<cl::sycl::info::kernel::num_args>();
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_info__ */
