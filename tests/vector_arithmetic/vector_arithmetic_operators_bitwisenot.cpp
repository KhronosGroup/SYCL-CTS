
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_arithmetic_operators.py
//
************************************************************************************/
/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:  (c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#include "./../../util/math_helper.h"
#include "./../../util/type_names.h"

#define TEST_NAME vector_arithmetic_operators_bitwisenot
#define KERNEL_NAME cKernel_vector_arithmetic_operators_bitwisenot
#define ARITHMETIC_OPERATOR ~
#define COMPOUND_OPERATION 0
#define NAMESPACE vector_arithmetic_operators_bitwisenot__

namespace NAMESPACE
{
using namespace sycl_cts;
using namespace cl::sycl;

/** kernel functor
 */
template <typename T>
class KERNEL_NAME
{
protected:
    typedef accessor<T, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> t_readAccess;
    typedef accessor<T, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> t_writeAccess;

    t_writeAccess m_o; /* output     */
    t_readAccess m_x;  /* argument X */
    t_readAccess m_y;  /* argument Y */

public:
    KERNEL_NAME( t_writeAccess out_, t_readAccess x_, t_readAccess y_ )
        : m_o( out_ )
        , m_x( x_ )
        , m_y( y_ )
    {
    }

    void operator()( item<1> item )
    {
        auto &o = m_o[item.get_global()];
        auto  x = m_x[item.get_global()];
        auto  y = m_y[item.get_global()];
        if ( COMPOUND_OPERATION )
        {
            o ARITHMETIC_OPERATOR x;
        }
        else
        {
            o = x ARITHMETIC_OPERATOR y;
        }
    }
};

/** test SYCL header for compilation
 */
template <typename T>
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info &out ) const
    {
        using sycl_cts::util::STRING;
        STRING name = STRING( TOSTRING( TEST_NAME ) ) + "_" + type_name<T>( );
        set_test_info( out, name.c_str( ), TEST_FILE );
    }

    /** execute this test
     *  @return, one of test_result enum
     */
    virtual void run( util::logger &log )
    {
        try
        {
            T xdata;
            math::fill( xdata, 6 );
            T ydata;
            math::fill( ydata, 3 );
            T odata;
            math::fill( odata, 3 );

            // construct the cts default selector
            cts_selector selector;

            /* create command queue */
            queue l_queue( selector );
            {
              buffer<T, 1> xbuf( &xdata, range<1> ( math::numElements( xdata ) ) );
              buffer<T, 1> ybuf( &ydata, range<1> ( math::numElements( ydata ) ) );
              buffer<T, 1> obuf( &odata, range<1> ( math::numElements( odata ) ) );

              /* add command to queue */
              l_queue.submit( [&]( handler& cgh )
              {
                  auto xptr = xbuf.template get_access<cl::sycl::access::mode::read>( cgh );
                  auto yptr = ybuf.template get_access<cl::sycl::access::mode::read>( cgh );
                  auto optr = obuf.template get_access<cl::sycl::access::mode::write>( cgh );

                  /* instantiate the kernel */
                  auto kern = KERNEL_NAME<T>(optr, xptr, yptr);

                  /* execute the kernel */
                  cgh.parallel_for(
                  nd_range<1>( range<1>( 1 ), range<1>( 1 ) ), kern );

               } );
            }

            if ( COMPOUND_OPERATION )
            {
                ydata ARITHMETIC_OPERATOR xdata;
                if ( all(odata != ydata) )
                {
                    FAIL( log, "results don't match" );
                }
            }
            else
            {
                if ( all( odata != ( xdata ARITHMETIC_OPERATOR ydata ) ) )
                {
                    FAIL( log, "results don't match" );
                }
            }

            l_queue.wait_and_throw();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }

    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME<float2>> proxy2;
util::test_proxy<TEST_NAME<float3>> proxy3;
util::test_proxy<TEST_NAME<float4>> proxy4;
util::test_proxy<TEST_NAME<float8>> proxy8;
util::test_proxy<TEST_NAME<float16>> proxy16;

}; /* namespace NAMESPACE */
