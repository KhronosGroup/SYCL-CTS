/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#define SYCL_SIMPLE_SWIZZLES
#include "../common/common.h"
#include "../../oclmath/mt19937.h"

#define TEST_NAME math_relational_ternary

// gentype bitselect (gentype a, gentype b, gentype     c);
// gentype select    (gentype a, gentype b, igeninteger c);
// gentype select    (gentype a, gentype b, ugeninteger c);

namespace math_relational_ternary__
{
using namespace sycl_cts;

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

/* generate a new parameter */
template <typename type_t>
type_t next_param(  )
{
    
    return type_t( 0 );
}

/* verify result for given parameters */
template <typename scl_t, typename vec_t>
bool verify_func( scl_t result, vec_t p1, vec_t p2, vec_t p3 )
{
    return true;
}

/**
 */
template <typename type_t>
struct test_class
{
    static const uint32_t num_params_k  = 3;
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
                (m_params[j])[i] = next_param<type_t>(  );
    }

    void execute( util::logger & log, cl::sycl::queue & sycl_queue )
    {
        cl::sycl::buffer<type_t, 1> buf_output ( & m_output[0]    , cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param_1( &(m_params[0])[0], cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param_2( &(m_params[1])[0], cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param_3( &(m_params[2])[0], cl::sycl::range<1>( buffer_size_k ) );

        sycl_queue.submit( [&]( cl::sycl::handler & cgh )
        {
            auto acc_output  = buf_output .template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param_1 = buf_param_1.template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param_2 = buf_param_2.template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param_3 = buf_param_3.template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );

            cgh.parallel_for<test_class>( cl::sycl::range<1>( buffer_size_k ), [=]( cl::sycl::id<1> id )
            {
                type_t & out = acc_output [id];
                type_t & pr1 = acc_param_1[id];
                type_t & pr2 = acc_param_2[id];
                type_t & pr3 = acc_param_3[id];

                out = cl::sycl::bitselect( pr1, pr2, pr3 );
                out = cl::sycl::select   ( pr1, pr2, pr3 );
            } );
        } );
    }

    void verify( util::logger & log )
    {
        for ( uint32_t i = 0; i < buffer_size_k; i++ ) {
            if (! verify_func( m_output[i], (m_params[0])[i], (m_params[1])[i], (m_params[2])[i] ) )
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

            test_class<cl::sycl::float3> test_float3;
            test_float3.run( log );

            test_class<cl::sycl::float4> test_float4;
            test_float4.run( log );

            queue.wait_and_throw();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace math_relational_ternary__ */
