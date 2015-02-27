/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_as_templated_functor

namespace kernel_as_templated_functor__
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
            auto access_input  = buffer_input.template  get_access<cl::sycl::access::read>( cgh );
            auto access_output = buffer_output.template get_access<cl::sycl::cl::sycl::access::mode::write>( cgh );
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


            
            if ( !test_kernel_functor<float>( 42.f, log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<double>( 42.0, log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<int8_t>( int8_t( 0xff ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<int16_t>( int16_t( 0xffff ), log, sycl_queue ) )
                return;
            
            if ( !test_kernel_functor<int32_t>( int32_t( 0xffffffff ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<int64_t>( int64_t( ~0ull ), log, sycl_queue ) )
                return;
                                    
            if ( !test_kernel_functor<uint8_t>( uint8_t( 0xffu ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<uint16_t>( uint16_t( 0xffffu ), log, sycl_queue ) )
                return;
            
            if ( !test_kernel_functor<uint32_t>( uint32_t( 0xffffffffu ), log, sycl_queue ) )
                return;

            if ( !test_kernel_functor<uint64_t>( uint64_t( ~0ull ), log, sycl_queue ) )
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

} /* namespace kernel_as_templated_functor__ */
