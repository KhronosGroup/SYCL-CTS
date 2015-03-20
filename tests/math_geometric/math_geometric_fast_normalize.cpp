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
#include "./../../util/math_helper.h"

#define TEST_NAME math_geometric_fast_normalize

namespace TEST_NAMESPACE
{
using namespace sycl_cts;

/* list of special edge cases to test stored as floating
 * point bit patterns.
 *
 * each value in this list will be used for both
 * x and y arguments to a math function.
 *
 * for ( int i=0; i<nEdgeCases*nEdgeCases; i++ )
 * {
 *     xarg = gEdgeCases[ i % nEdgeCases ];
 *     yarg = gEdgeCases[ i / nEdgeCases ];
 *     ...
 * }
 */
uint32_t gEdgeCases[] = { 0x7fc00000, 0xff800000, 0xff7fffff, 0xdf800001, 0xdf800000, 0xdf7fffff, 0xdf000001, 0xdf000000, 0xdeffffff,
                          0xcf800001, 0xcf800000, 0xcf7fffff, 0xcf000001, 0xcf000000, 0xceffffff, 0xc47a0000, 0xc2c80000, 0xc0800000,
                          0xc0600000, 0xc0400000, 0xc0400001, 0xc0200000, 0xc03fffff, 0xc0000000, 0xbfc00001, 0xbfc00000, 0xbfbfffff,
                          0xbf800001, 0xbf800000, 0xbf7fffff, 0xbf000001, 0xbf000000, 0xbeffffff, 0xbe800001, 0xbe800000, 0xbe7fffff,
                          0x80800001, 0x80800000, 0x807fffff, 0x800007ff, 0x8000007f, 0x80000007, 0x80000006, 0x80000005, 0x80000004,
                          0x80000003, 0x80000002, 0x80000001, 0x80000000, 0xffc00000, 0x7f800000, 0x7f7fffff, 0x5f800001, 0x5f800000,
                          0x5f7fffff, 0x5f000001, 0x5f000000, 0x5effffff, 0x4f800001, 0x4f800000, 0x4f7fffff, 0x4f000001, 0x4f000000,
                          0x4effffff, 0x447a0000, 0x42c80000, 0x40800000, 0x40600000, 0x40400000, 0x40400001, 0x40200000, 0x403fffff,
                          0x40000000, 0x3fc00001, 0x3fc00000, 0x3fbfffff, 0x3f800001, 0x3f800000, 0x3f7fffff, 0x3f000001, 0x3f000000,
                          0x3effffff, 0x3e800001, 0x3e800000, 0x3e7fffff, 0x00800001, 0x00800000, 0x007fffff, 0x000007ff, 0x0000007f,
                          0x00000007, 0x00000006, 0x00000005, 0x00000004, 0x00000003, 0x00000002, 0x00000001, 0x00000000 };
const uint32_t nEdgeCases = sizeof( gEdgeCases ) / sizeof( uint32_t );


template <typename T, int n>
T length( cl::sycl::vec<T, n> v )
{
    T total = 0.0;
    for ( int i = 0; i < n; ++i )
        total += ::getElement(v,i);
    return sqrt( total );
}

template <typename T, int n>
cl::sycl::vec<T, n> normalize( cl::sycl::vec<T, n> v )
{
    cl::sycl::vec<T, n> ret;
    T len = length( v );
    for ( int i = 0; i < n; ++i )
        ::setElement(ret,i,(::getElement(v,i) / len));
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
bool verify_func( type_t & input, type_t & param )
{
  float ulpTolerance = 8192.f;

  cl::sycl::float4 res = normalize( param  );
  float errorTolerance;

  errorTolerance = ulpTolerance * fmaxf ( fmaxf( ::getElement(param,1) , ::getElement(param,1) ), ::getElement(param,2)) ;

  cl::sycl::float3 errs;
  for ( int i = 0; i < 3; ++i )
  {
      ::setElement(errs,i, fabsf( ::getElement(input,i) - ::getElement(res,i)));
      if ( ::getElement(errs,i) > errorTolerance )
          return false;
  }

  return true;
}
/**
 */
template <typename type_t>
struct test_class
{
#if WIMPY_MODE
    /* wimpy mode constant */
    static const uint32_t nScale = 0x100;
#else
    /* full (brutal) mode constant */
    static const uint32_t nScale = 0x1;
#endif

    /* size of the host side buffer */
    static const uint32_t nBufferSize = 0x1000;

    /* size of the device side buffer */
    static const uint32_t nSubBufferSize = 0x100;

    static const uint32_t buffer_size_k = 1024;

    test_buffer<type_t, buffer_size_k> m_param;
    test_buffer<type_t, buffer_size_k> m_output;

    test_class( )
        : m_param()
    {
    }

    uint64_t m_index;
    uint64_t m_edge_index;

    /* record the max ULP error */
    float m_max_ulp;

    MTdata m_randData;


    /** return a mask with bits being set
     */
    uint64_t type_mask( ) {

        if ( typeid( type_t::elem_t ) == typeid( float ) )
            return ~( ( 1ull << 32 ) - 1 );

#if SYCL_CTS_TEST_DOUBLE
        if ( typeid( type_t::elem_t ) == typeid( double ) )
            return ~( ( 1ull << 32 ) - 1 );
#endif
        return 0;
    }

    /**
     */
    typename type_t::elem_t generate_scalar( uint32_t x )
    {
        return type_t::elem_t( math::int_to_float( x ) );
    }


    void generate( )
    {
          const uint32_t nElms = type_length(type_t( ) );

          /* fill data buffer */
          for ( uint32_t i = 0; i < nBufferSize; i++ )
          {
              /* access buffers */
             type_t & e_param1 = m_param[ i ];
             type_t & e_output = m_output [ i ];

              /* for each element */
              for ( uint32_t j = 0; j < nElms; j++ )
              {
                  /* truncate the index value to type range */
                  uint64_t ix = uint64_t( m_index & ~type_mask() );

                  if ( m_edge_index < (nEdgeCases * nEdgeCases) )
                  {
                      ix = m_edge_index++;
                      /* use the edge case list */
                      type_set( e_param1, j, generate_scalar( gEdgeCases[ ix % nEdgeCases ] ) );
                  }
                  else
                  {
                      /* generate new parameter value */
                      type_set( e_param1, j, generate_scalar( genrand_int32( m_randData ) ) );
                  }

                  m_index += nScale;

                  /* clear the output buffer */
                  type_set( e_output, j, type_t::elem_t( 1.f ) );
              }
          }
}

    void execute( util::logger & log, cl::sycl::queue & sycl_queue )
    {
        cl::sycl::buffer<type_t, 1> buf_output( &m_output[0], cl::sycl::range<1>( buffer_size_k ) );
        cl::sycl::buffer<type_t, 1> buf_param ( &m_param [0], cl::sycl::range<1>( buffer_size_k ) );

        sycl_queue.submit( [&]( cl::sycl::handler & cgh )
        {
            auto acc_output = buf_output.template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>( cgh );
            auto acc_param  = buf_param .template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>( cgh );

            cgh.parallel_for<test_class>( cl::sycl::range<1>( buffer_size_k ), [=]( cl::sycl::id<1> id )
            {
                type_t & out = acc_output[id];
                type_t & prm = acc_param [id];

                out = cl::sycl::fast_normalize( prm );
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

            test_class<cl::sycl::float3> test_float3;
            test_float3.run( log );

            test_class<cl::sycl::float4> test_float4;
            test_float4.run( log );

#ifdef SYCL_CTS_TEST_DOUBLE

            test_class<cl::sycl::double3> test_double3;
            test_double3.run( log );

            test_class<cl::sycl::double4> test_double4;
            test_double4.run( log );

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

} /* namespace math_geometric_fast_normalize__ */
