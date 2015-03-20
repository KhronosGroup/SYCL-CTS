/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../../oclmath/mt19937.h"
#include "./../../util/math_vector.h"

#define TEST_NAME math_geometric_distance

namespace TEST_NAMESPACE
{
using namespace sycl_cts;

template <typename T, int n>
T distance( cl::sycl::vec<T, n> v1, cl::sycl::vec<T, n> v2 )
{
    T total = 0;
    for ( int i = 0; i < n; ++i )
        total += (::getElement(v2,i) - ::getElement(v1,i) * (::getElement(v2,i) - ::getElement(v1,i)));
    return sqrt( total );
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
type_t next_param( /* TODO: pass in RNG */ )
{

    return type_t( 0 );
}

bool verify_scalar( float rs, float px, float py )
{

    return true;
}

// NOTE: Have two verifiers, one for double, one for float, this one is for float
template <typename type_t>
bool verify_vector( type_t & input, type_t & param_x, type_t & param_y )
{
    uint32_t num_elms = sizeof( type_t ) / sizeof( float );

    for ( uint32_t i = 0; i < num_elms; i++ )
    {
        if ( !verify_scalar( ::getElement(input,i), ::getElement(param_x,i), ::getElement(param_y,i) ) )
            return false;
    }

    return true;
}

/**
 */
template <typename type_t>
struct test_class
{
    static const uint32_t num_params_k  = 2;
    static const uint32_t buffer_size_k = 1024;

    test_buffer<type_t, buffer_size_k> m_params[ num_params_k ];
    test_buffer<type_t, buffer_size_k> m_output;

    test_class( )
        : m_params()
    {
    }

    void generate( )
    {
        for ( uint32_t i = 0; i < buffer_size_k; i++ )
            for ( int j = 0; j < num_params_k; j++ )
                (m_params[j])[i] = next_param<type_t>( /* TODO: pass in RNG */ );
    }

    void execute( util::logger & log, cl::sycl::queue & sycl_queue )
    {
        cl::sycl::buffer<type_t, 1> buf_output ( & m_output[0]    , cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param_1( &(m_params[0])[0], cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param_2( &(m_params[1])[0], cl::sycl::range<1>( buffer_size_k ) );

        sycl_queue.submit( [&]( cl::sycl::handler & cgh )
        {
            auto acc_output  = buf_output .template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param_1 = buf_param_1.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param_2 = buf_param_2.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>( cgh );

            cgh.parallel_for<test_class>( cl::sycl::range<1>( buffer_size_k ), [=]( cl::sycl::id<1> id )
            {
                type_t & out = acc_output [id];
                type_t & pr1 = acc_param_1[id];
                type_t & pr2 = acc_param_2[id];

                out = cl::sycl::distance( pr1, pr2 );
            } );
        } );
    }

    void verify( util::logger & log )
    {
        for ( uint32_t i = 0; i < buffer_size_k; i++ ) {
            if (! verify_vector( m_output[i], (m_params[0])[i], (m_params[1])[i] ) )
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

            {
                test_class<cl::sycl::float2> testf2;
                testf2.run( log );
            }
            {
                test_class<cl::sycl::float3> testf3;
                testf3.run( log );
            {
            }
                test_class<cl::sycl::float4> testf4;
                testf4.run( log );
            }

#ifdef SYCL_CTS_TEST_DOUBLE

            {
                test_class<cl::sycl::double2> testf2;
                testf2.run( log );
            }
            {
                test_class<cl::sycl::double3> testf3;
                testf3.run( log );
            {
            }
                test_class<cl::sycl::double4> testf4;
                testf4.run( log );
            }

#endif
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

} /* namespace math_geometric_distance__ */
