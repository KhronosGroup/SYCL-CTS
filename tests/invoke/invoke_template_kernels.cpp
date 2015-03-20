/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME invoke_template_kernels

namespace invoke_template_kernels__
{
using namespace sycl_cts;
using namespace cl::sycl;

template<typename T>
class templated_functor
{
    typedef accessor<T, 1, cl::sycl::access::mode::read , cl::sycl::access::target::global_buffer> read_t;
    typedef accessor<T, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> write_t;

    read_t  m_in;
    write_t m_out;

public:
    templated_functor( read_t in, write_t out )
        : m_in ( in  )
        , m_out( out )
    {
    }

    void operator() ()
    {
        m_out[0] = m_in[0];
    }
};

template <typename T>
bool test_kernel_functor( T in_value, util::logger &log, queue & sycl_queue )
{
    T input = in_value, output = 0;
    {
        cl::sycl::buffer<T,1> buffer_input ( &input,  cl::sycl::range<1>(1));
        cl::sycl::buffer<T,1> buffer_output( &output, cl::sycl::range<1>(1));
        sycl_queue.submit( [&]( handler& cgh )
        {
            auto access_input  = buffer_input.template  get_access<cl::sycl::access::mode::read>( cgh );
            auto access_output = buffer_output.template get_access<cl::sycl::access::mode::write>( cgh );
            templated_functor<T> kernel( access_input, access_output );
            cgh.single_task( kernel );
        } );
    }
    return CHECK_VALUE( log, input, output, 0 );
}

/** test cl::sycl::kernel from functor
 */
class TEST_NAME : public sycl_cts::util::test_base
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
            queue sycl_queue( selector );

            static const float test_float_value = 10;
            static const double test_double_value = 10;
            
            if ( !test_kernel_functor<float>( test_float_value, log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<double>( test_double_value, log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<int8_t>( int8_t( INT8_MAX ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<int16_t>( int16_t( INT16_MAX ), log, sycl_queue ) )
                return;
            
            if ( !test_kernel_functor<int32_t>( int32_t( INT32_MAX ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<int64_t>( int64_t( INT64_MAX ), log, sycl_queue ) )
                return;
                                    
            if ( !test_kernel_functor<uint8_t>( uint8_t( UINT8_MAX ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<uint16_t>( uint16_t( UINT16_MAX ), log, sycl_queue ) )
                return;
            
            if ( !test_kernel_functor<uint32_t>( uint32_t( UINT32_MAX ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<uint64_t>( uint64_t( UINT64_MAX ), log, sycl_queue ) )
                return;

            sycl_queue.wait_and_throw();

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

} /* namespace invoke_template_kernels__ */
