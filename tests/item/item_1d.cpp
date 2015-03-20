/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME item_1d

namespace test_item_1d__
{
using namespace sycl_cts;
using namespace cl::sycl;

class kernel_item_1d
{
protected:
    typedef accessor<int, 1, cl::sycl::access::mode::read,  cl::sycl::access::target::global_buffer> t_readAccess;
    typedef accessor<int, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> t_writeAccess;
    
    t_readAccess  m_x;
    t_writeAccess m_o;

public:
    kernel_item_1d( t_readAccess in_, t_writeAccess out_ )
        : m_x( in_  )
        , m_o( out_ )
    {
    }

    void operator()( item<1> item )
    {
        id<1> gid = item.get_global( );

        size_t dim_a = item.get( 0 );
        size_t dim_b = item[0];

        range<1> globalRange = item.get_global_range( );
        range<1> localRange  = item.get_local_range( );

        size_t group = item.get_group( 0 );

        id<1> offset = item.get_offset( );

        /* get work item range */
        const size_t nWidth  = globalRange.get( )[0];

        /* find the array id for this work item */
        size_t index = gid.get( )[0]; /* x */

        /* get the global linear id */
        const size_t glid = item.get_global_linear();
        
        /* compare against the precomputed index */
        m_o[int(glid)] = (m_x[int(glid)] == static_cast<int>(index));
    }
};

void buffer_fill( int *buf, const int nWidth )
{
    for ( int i = 0; i < nWidth; i++ )
        buf[i] = i;
}

int buffer_verify( int *buf, const int nWidth )
{
    int nErrors = 0;
    for ( int i = 0; i < nWidth; i++ )
        nErrors += (buf[i] == 0);
    return nErrors;
}

bool test_item_1d( util::logger & log )
{
    const int nWidth = 64;

    /* allocate host buffers */
    std::unique_ptr<int> dataIn ( new int[ nWidth ] );
    std::unique_ptr<int> dataOut( new int[ nWidth ] );

    /* clear host buffers */
    memset( dataIn.get() , 0, nWidth * sizeof( int ) );
    memset( dataOut.get(), 0, nWidth * sizeof( int ) );

    /*  */
    buffer_fill( dataIn.get( ), nWidth );

    try
    {
        range<1> dataRange( nWidth );

        buffer<int, 1> bufIn ( dataIn .get(), dataRange );
        buffer<int, 1> bufOut( dataOut.get(), dataRange );

        cts_selector selector;
        queue cmdQueue( selector );

        cmdQueue.submit( [&]( handler& cgh )
        {
            auto accIn  = bufIn  .template get_access<cl::sycl::access::mode::read >( cgh );
            auto accOut = bufOut .template get_access<cl::sycl::access::mode::write>( cgh );

            kernel_item_1d kern = kernel_item_1d( accIn, accOut );
            cgh.parallel_for( nd_range<1>( dataRange, range<1>( nWidth / 8 ) ), kern );
        });

        cmdQueue.wait_and_throw();

    }
    catch ( cl::sycl::exception e )
    {
        log_exception( log, e );
        FAIL( log, "sycl exception caught" );
        return false;
    }

    /*  */
    if ( buffer_verify( dataOut.get(), nWidth ) )
    {
        FAIL( log, "item incorrectly mapped" );
        return false;
    }

    return true;
}

/** test cl::sycl::device initialization
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
        test_item_1d( log );
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_item_1d__ */
