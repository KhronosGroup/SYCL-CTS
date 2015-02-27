/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../../oclmath/mt19937.h"

#define TEST_NAME math_geometric_normalize

namespace TEST_NAMESPACE
{
using namespace sycl_cts;

template <typename T, int n>
T length( cl::sycl::vec<T, n> v )
{
    T len = 0.0;
    for ( int i = 0; i < n; ++i )
        total += v[i];
    return sqrt( total );
}

template <typename T, int n>
cl::sycl::vec<T, n> normalize( cl::sycl::vec<T, n> v )
{
    cl::sycl::vec<T, n> ret;
    T len = length( v );
    for ( int i = 0; i < n; ++i )
        ret[i] = v[i] / len;
    return ret;
}

/**
 */
template <typename type_t, uint32_t size>
struct test_buffer
{
    type_t m_buffer[ size ];

    test_buffer( )
    {
        memset( m_buffer, 0, sizeof( m_buffer ) );
    }

    type_t & operator [] ( uint32_t ix )
    {
        assert( ix >= 0 && ix < size );
        return m_buffer[ix];
    }
};

template <typename type_t>
type_t next_param(  )
{
    
    return type_t( 0 );
}

template <typename type_t>
bool verify_func( type_t & input, type_t & param )
{
    return true;
}

/**
 */
template <typename type_t>
struct test_class
{
    static const uint32_t buffer_size_k = 1024;

    test_buffer<type_t, buffer_size_k> m_param;
    test_buffer<type_t, buffer_size_k> m_output;

    test_class( )
        : m_param()
    {
    }

    void generate( )
    {
        for ( uint32_t i = 0; i < buffer_size_k; i++ )
            m_param[i] = next_param<type_t>(  );
    }

    void execute( util::logger & log, cl::sycl::queue & sycl_queue )
    {
        cl::sycl::buffer<type_t, 1> buf_output( &m_output[0], cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param ( &m_param [0], cl::sycl::range<1>( buffer_size_k ) );

        sycl_queue.submit( [&]( cl::sycl::handler & cgh )
        {
            auto acc_output = buf_output.template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param  = buf_param .template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );

            cgh.parallel_for<test_class>( cl::sycl::range<1>( buffer_size_k ), [=]( cl::sycl::id<1> id )
            {
                type_t & out = acc_output[id];
                type_t & prm = acc_param [id];

                out = cl::sycl::normalize( prm );
            } );
        } );
    }

    void verify( util::logger & log )
    {
        for ( uint32_t i = 0; i < buffer_size_k; i++ ) {
            if (! verify_func( m_output[i], m_param[i] ) )
            {
                FAIL( log, "verification failed" );
                break;
            }
        }
    }

    void run( util::logger & log )
    {
        const uint32_t num_itts = 100;

        cl::sycl::queue sycl_queue;

        for ( int i = 0; i < num_itts; i++ )
        {
            generate( );
            execute( log, sycl_queue );
            verify( log );

            if ( log.has_failed( ) )
                break;
        }
    }
    
    bool setup( )
    {
        return true;
    }

    void cleanup( )
    {
    }
};

/** 
 */
class TEST_NAME : public util::test_base
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
        using namespace cl::sycl;

        try
        {
            cl::sycl::queue sycl_queue;

            test_class<cl::sycl::float3>  test_float3;
            test_float3.run( log );

            test_class<cl::sycl::float4>  test_float4;
            test_float4.run( log );

            test_class<cl::sycl::double3> test_double3;
            test_double3.run( log );

            test_class<cl::sycl::float4>  test_double4;
            test_double4.run( log );

            sycl_queue.wait_and_throw();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace math_geometric_normalize__ */
