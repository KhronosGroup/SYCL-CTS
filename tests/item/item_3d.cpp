/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME item_3d

namespace test_item_3d__
{
using namespace sycl_cts;
using namespace cl::sycl;

class kernel_item_3d
{
protected:
    typedef accessor<int, 3, cl::sycl::access::mode::read,  cl::sycl::access::target::global_buffer> t_readAccess;
    typedef accessor<int, 3, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> t_writeAccess;
    
    t_readAccess  m_x;
    t_writeAccess m_o;

public:
    kernel_item_3d( t_readAccess in_, t_writeAccess out_ )
        : m_x( in_  )
        , m_o( out_ )
    {
    }

    void operator()( item<3> item )
    {
        id<3> gid = item.get_global( );

        size_t dim_a = item.get( 0 );
        size_t dim_b = item[0];

        range<3> globalRange = item.get_global_range( );
        range<3> localRange  = item.get_local_range( );

        size_t group = item.get_group( 0 );

        id<1> offset = item.get_offset( );

        /* get work item range */
        const size_t nWidth  = globalRange.get( )[0];
        const size_t nHeight = globalRange.get( )[1];
        const size_t nDepth  = globalRange.get( )[2];
        
        /* find the array id for this work item */
        size_t index = 
             gid.get()[0] +                         /* x */
            (gid.get()[1] * nWidth) +               /* y */
            (gid.get()[2] * nWidth * nHeight);      /* z */
        
        /* compare against the precomputed index */
        m_o[item] = (m_x[item] == static_cast<int>(index));
    }
};

void buffer_fill( int *buf, const int nWidth, const int nHeight, const int nDepth )
{
    for ( int k = 0; k < nDepth; k++ )
    {
        for ( int j = 0; j < nHeight; j++ )
        {
            for ( int i = 0; i < nWidth; i++ )
            {
                const int index = (k * nWidth * nHeight) + (j * nWidth) + i;
                buf[index] = index;
            }
        }
    }
}

int buffer_verify( int *buf, const int nWidth, const int nHeight, const int nDepth )
{
    int nErrors = 0;
    for ( int i = 0; i < nWidth * nHeight * nDepth; i++ )
    {
        nErrors += (buf[i] == 0);
    }
    return nErrors;
}

bool test_item_3d( util::logger & log )
{
    const int nWidth  = 64;
    const int nHeight = 64;
    const int nDepth  = 64;

    const int nSize = nWidth * nHeight * nDepth;

    /* allocate host buffers */
    std::unique_ptr<int> dataIn ( new int[ nSize ] );
    std::unique_ptr<int> dataOut( new int[ nSize ] );

    /* clear host buffers */
    memset( dataIn.get() , 0, nSize * sizeof( int ) );
    memset( dataOut.get(), 0, nSize * sizeof( int ) );

    buffer_fill( dataIn.get( ), nWidth, nHeight, nDepth );

    try
    {
        range<3> dataRange( nWidth, nHeight, nDepth );

        buffer<int, 3> bufIn ( dataIn .get(), dataRange );
        buffer<int, 3> bufOut( dataOut.get(), dataRange );

        cts_selector selector;
        queue cmdQueue( selector );

        cmdQueue.submit( [&]( handler& cgh )
        {
            auto accIn  = bufIn  .template get_access<cl::sycl::access::mode::read >( cgh );
            auto accOut = bufOut .template get_access<cl::sycl::access::mode::write>( cgh );

            kernel_item_3d kern = kernel_item_3d( accIn, accOut );

            auto r = nd_range<3>( dataRange, range<3>( 1, 1, 1 ) );
            cgh.parallel_for( r, kern );
        });

        cmdQueue.wait_and_throw();

    }
    catch ( cl::sycl::exception e )
    {
        log_exception( log, e );
        FAIL( log, "sycl exception caught" );
        return false;
    }
    
    if ( buffer_verify( dataOut.get(), nWidth, nHeight, nDepth ))
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
        test_item_3d( log );
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_item_2d__ */
