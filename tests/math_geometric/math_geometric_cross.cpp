/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../../oclmath/mt19937.h"

#define TEST_NAME math_geometric_cross

namespace math_geometric_cross__
{
using namespace sycl_cts;

cl::sycl::float3  cross( cl::sycl::float3  p0, cl::sycl::float3  p1 )
{
    float x, y, z;
    x = p0.y() * p1.z() - p0.z() * p1.y();
    y = p0.z() * p1.x() - p0.x() * p1.z();
    z = p0.x() * p1.y() - p0.y() * p1.x();

    return { x, y, z };
}

cl::sycl::float4  cross( cl::sycl::float4  p0, cl::sycl::float4  p1 )
{
    float x, y, z;
    x = p0.y() * p1.z() - p0.z() * p1.y();
    y = p0.z() * p1.x() - p0.x() * p1.z();
    z = p0.x() * p1.y() - p0.y() * p1.x();

    return { x, y, z, 0 };
}

cl::sycl::double3 cross( cl::sycl::double3 p0, cl::sycl::double3 p1 )
{
    double x, y, z;
    x = p0.y() * p1.z() - p0.z() * p1.y();
    y = p0.z() * p1.x() - p0.x() * p1.z();
    z = p0.x() * p1.y() - p0.y() * p1.x();

    return { x, y, z };
}

cl::sycl::double4 cross( cl::sycl::double4 p0, cl::sycl::double4 p1 )
{
    double x, y, z;
    x = p0.y() * p1.z() - p0.z() * p1.y();
    y = p0.z() * p1.x() - p0.x() * p1.z();
    z = p0.x() * p1.y() - p0.y() * p1.x();

    return { x, y, z, 0 };
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

bool verify_func( cl::sycl::float4 & input, cl::sycl::float4 & param_x, cl::sycl::float4 & param_y )
{
    
    return true;
}

bool verify_func( cl::sycl::float3 & input, cl::sycl::float3 & param_x, cl::sycl::float3 & param_y )
{
    
    return true;
}

bool verify_func( cl::sycl::double4 & input, cl::sycl::double4 & param_x, cl::sycl::double4 & param_y )
{
    
    return true;
}

bool verify_func( cl::sycl::double3 & input, cl::sycl::double3 & param_x, cl::sycl::double3 & param_y )
{
    
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
                (m_params[j])[i] = next_param<type_t>(  );
    }

    void execute( util::logger & log, cl::sycl::queue & sycl_queue )
    {
        cl::sycl::buffer<type_t, 1> buf_output ( & m_output[0]    , cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param_1( &(m_params[0])[0], cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param_2( &(m_params[1])[0], cl::sycl::range<1>( buffer_size_k ) );

        sycl_queue.submit( [&]( cl::sycl::handler & cgh )
        {
            auto acc_output  = buf_output .template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param_1 = buf_param_1.template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param_2 = buf_param_2.template get_access<cl::sycl::cl::sycl::access::target::global_buffer>( cgh );

            cgh.parallel_for<test_class>( cl::sycl::range<1>( buffer_size_k ), [=]( cl::sycl::id<1> id )
            {
                type_t & out = acc_output [id];
                type_t & pr1 = acc_param_1[id];
                type_t & pr2 = acc_param_2[id];

                out = cl::sycl::cross( pr1, pr2 );
            } );
        } );
    }

    void verify( util::logger & log )
    {
        for ( uint32_t i = 0; i < buffer_size_k; i++ ) {
            if (! verify_func( m_output[i], (m_params[0])[i], (m_params[1])[i] ) )
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

            test_class<cl::sycl::double3> test_double3;
            test_double3.run( log );

            test_class<cl::sycl::float4> test_double4;
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

} /* namespace math_geometric_cross__ */
