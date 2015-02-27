/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_params_lambda

namespace kernel_params_lambda__
{
using namespace sycl_cts;

struct basic_type_t
{
    uint32_t a;
    uint8_t  b;
    uint8_t  c;
};

enum
{
    ref_a   = 1 ,
    ref_b_a = 10,
    ref_b_b = 12,
    ref_b_c = 5 ,
};

struct array_type_t {

    int32_t int_array[8];
};

/** 
 */
class TEST_NAME : public util::test_base
{
public:

    typedef cl::sycl::accessor<uint32_t, 1, cl::sycl::cl::sycl::access::mode::write, cl::sycl::cl::sycl::access::target::global_buffer> acc_uint32_t;

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
        using namespace cl::sycl;

        // 
        uint32_t result = 0;

        try
        {
            default_selector my_selector;
            queue   my_queue( my_selector );

            buffer<uint32_t> buf_result( &result, range<1>( 1 ) );

            my_queue.submit( [&]( cl::sycl::handler& cgh )
            {
                // access the output 
                auto acc_pass = buf_result. template get_access< cl::sycl::access::mode::write >( cgh );

                const uint32_t     var_a = ref_a;
                const basic_type_t var_b = { ref_b_a, ref_b_b, ref_b_c };

                array_type_t my_array;
                for ( uint32_t i = 0; i < 8; i++ )
                    my_array.int_array[i] = i;

                cgh.single_task<class test_kernel_name>( [=]( )
                {
                    uint32_t pass = 1;
                    pass &= (var_a   == ref_a  );
                    pass &= (var_b.a == ref_b_a);
                    pass &= (var_b.b == ref_b_b);
                    pass &= (var_b.c == ref_b_c);

                    for ( int32_t i = 0; i < 8; i++ )
                        pass &= ( my_array.int_array[i] == i );

                    acc_pass[0] = pass;
                } );
            } );

            my_queue.wait_and_throw();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }

        if ( result != 1 )
            FAIL( log, "incorrect values passed to lambda kernel" );

    }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_params_lambda__ */
