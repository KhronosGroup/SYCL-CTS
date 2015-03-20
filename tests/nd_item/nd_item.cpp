/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item

namespace test_nd_item__
{
using namespace sycl_cts;
using namespace cl::sycl;

template <int dimensions>
class kernel_nd_item
{
protected:
    typedef accessor<int, dimensions, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> t_readAccess;
    typedef accessor<int, dimensions, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> t_writeAccess;

    t_readAccess m_globalID;
    t_readAccess m_localID;
    t_writeAccess m_o;

public:
    kernel_nd_item( t_readAccess inG_, t_readAccess inL_, t_writeAccess out_ )
        : m_globalID( inG_ )
        , m_localID( inL_ )
        , m_o( out_ )
    {
    }

    void operator()( cl::sycl::nd_item<dimensions> myitem )
    {
        bool failed = false;

        /* test global ID*/
        id<dimensions> global_id = myitem.get_global();
        size_t globals[dimensions];

        for ( int i = 0; i < dimensions; ++i )
        {
            globals[i] = myitem.get_global( i );
            if ( globals[i] != global_id.get()[i] )
                failed = true;
        }

        id<dimensions> local_id = myitem.get_local();
        size_t locals[dimensions];

        for ( int i = 0; i < dimensions; ++i )
        {
            locals[i] = myitem.get_local( i );
            if ( locals[i] != local_id.get()[i] )
                failed = true;
        }

        /* test group ID*/
        id<dimensions> group_id = myitem.get_group();
        size_t groups[dimensions];

        for ( int i = 0; i < dimensions; ++i )
        {
            groups[i] = myitem.get_group( i );
            if ( groups[i] != group_id.get()[i] )
                failed = true;
        }

        /* test range*/
        range<dimensions> globalRange = myitem.get_global_range();

        size_t globalIndex = global_id.get()[0] + ( global_id.get()[1] * globalRange.get()[0] ) +
                             ( global_id.get()[2] * globalRange.get()[0] * globalRange.get()[1] );

        for ( int i = 0; i < dimensions; ++i )
        {
            if ( m_globalID[global_id] != globalIndex )
                failed = true;
        }

        range<dimensions> localRange = myitem.get_local_range();

        size_t localIndex = local_id.get()[0] + ( local_id.get()[1] * localRange.get()[0] ) +
                            ( local_id.get()[2] * localRange.get()[0] * localRange.get()[1] );

        for ( int i = 0; i < dimensions; ++i )
        {
            if ( m_localID[local_id] != localIndex )
                failed = true;
        }

        for ( int i = 0; i < dimensions; ++i )
        {
            if ( group_id.get()[i] != ( global_id.get()[i] / localRange.get()[i] ) )
                failed = true;
        }

        /* test number of groups*/
        id<dimensions> num_groups = myitem.get_num_groups();
        size_t nGroups[dimensions];

        for ( int i = 0; i < dimensions; ++i )
        {
            nGroups[i] = myitem.get_num_groups( i );
            if ( nGroups[i] != num_groups.get()[i] )
                failed = true;
        }

        for ( int i = 0; i < dimensions; ++i )
        {
            size_t ratio = globalRange.get()[i] / localRange.get()[i];
            if ( ratio != num_groups.get()[i] )
                failed = true;
        }

        /* test NDrange and offset*/
        id<dimensions> offset = myitem.get_offset();
        nd_range<dimensions> NDRange = myitem.get_nd_range();

        range<dimensions> ndGlobal = NDRange.get_global_range();
        range<dimensions> ndLocal = NDRange.get_local_range();
        id<dimensions> ndOffset = NDRange.get_offset();

        for ( int i = 0; i < dimensions; ++i )
        {
            bool are_same = true;
            are_same &= globalRange.get()[i] == ndGlobal.get()[i];
            are_same &= localRange.get()[i] == ndLocal.get()[i];
            are_same &= offset.get()[i] == ndOffset.get()[i];

            failed = !are_same ? true : failed;
        }

        /* write back success or failure*/
        m_o[global_id] = failed ? 0 : 1;
    }
};

/* test that kernel returns expected data, i.e. all 1s*/
int check_nd_item( int* buf, const int nWidth, const int nHeight, const int nDepth )
{
    int nErrors = 0;
    for ( int i = 0; i < nWidth * nHeight * nDepth; i++ )
    {
        nErrors += ( buf[i] == 0 );
    }
    return nErrors;
}

/* Fill buffers with global and local work item ids*/
void populate( int* globalBuf, int* localBuf, const int* localSize, const int* globalSize )
{
    for ( int k = 0; k < globalSize[2]; k++ )
    {
        for ( int j = 0; j < globalSize[1]; j++ )
        {
            for ( int i = 0; i < globalSize[0]; i++ )
            {
                const int globIndex = ( k * globalSize[0] * globalSize[1] ) + ( j * globalSize[0] ) + i;
                globalBuf[globIndex] = globIndex;

                int local_i = i % localSize[0];
                int local_j = j % localSize[1];
                int local_k = k % localSize[2];

                const int locIndex = ( local_k * localSize[0] * localSize[1] ) + ( local_j * localSize[0] ) + local_i;
                localBuf[globIndex] = locIndex;
            }
        }
    }
}

void test_item( util::logger& log, cl::sycl::queue& queue )
{
    /* set sizes*/
    const int globalSize[3] = { 16, 16, 16 };
    const int localSize[3] = { 4, 4, 4 };
    const int nSize = globalSize[0] * globalSize[1] * globalSize[2];

    /* allocate host buffers */
    std::unique_ptr<int> globalIDs( new int[nSize] );
    std::unique_ptr<int> localIDs( new int[nSize] );
    std::unique_ptr<int> dataOut( new int[nSize] );

    /* set host buffers */
    populate( globalIDs.get(), localIDs.get(), localSize, globalSize );
    memset( dataOut.get(), 0, nSize * sizeof( int ) );

    /* create ranges*/
    range<3> globalRange( globalSize[0], globalSize[1], globalSize[2] );
    range<3> localRange( localSize[0], localSize[1], localSize[2] );
    nd_range<3> dataRange( globalRange, localRange );

    /* test 1 Dimension*/
    {
        buffer<int, 1> bufGlob( globalIDs.get(), globalRange );
        buffer<int, 1> bufLoc( localIDs.get(), globalRange );
        buffer<int, 1> bufOut( dataOut.get(), globalRange );

        queue.submit( [&]( handler& cgh )
        {
            auto accG = bufGlob.template get_access<cl::sycl::access::mode::read>( cgh );
            auto accL = bufLoc.template get_access<cl::sycl::access::mode::read>( cgh );
            auto accOut = bufOut.template get_access<cl::sycl::access::mode::write>( cgh );

            kernel_nd_item<1> kernel_1d( accG, accL, accOut );
            cgh.parallel_for( dataRange, kernel_1d );

        } );
    }
    /* check no errors are returned*/
    if ( check_nd_item( dataOut.get(), globalSize[0], 1, 1 ) )
    {
        FAIL( log, "item API inconsistency" );
        return;
    }

    /* test 2 Dimensions */
    memset( dataOut.get(), 0, nSize * sizeof( int ) );
    {
        buffer<int, 2> bufGlob( globalIDs.get(), globalRange );
        buffer<int, 2> bufLoc( localIDs.get(), globalRange );
        buffer<int, 2> bufOut( dataOut.get(), globalRange );

        queue.submit( [&]( handler& cgh )
        {
            auto accG = bufGlob.template get_access<cl::sycl::access::mode::read>( cgh );
            auto accL = bufLoc.template get_access<cl::sycl::access::mode::read>( cgh );
            auto accOut = bufOut.template get_access<cl::sycl::access::mode::write>( cgh );

            kernel_nd_item<2> kernel_2d( accG, accL, accOut );
            cgh.parallel_for( dataRange, kernel_2d );
        } );
    }
    /* check no errors are returned */
    if ( check_nd_item( dataOut.get(), globalSize[0], globalSize[1], 1 ) )
    {
        FAIL( log, "item API inconsistency" );
        return;
    }

    /* test 3 Dimensions */
    memset( dataOut.get(), 0, nSize * sizeof( int ) );
    {
        buffer<int, 3> bufGlob( globalIDs.get(), globalRange );
        buffer<int, 3> bufLoc( localIDs.get(), globalRange );
        buffer<int, 3> bufOut( dataOut.get(), globalRange );

        queue.submit( [&]( handler& cgh )
        {
            auto accG = bufGlob.get_access<cl::sycl::access::mode::read>( cgh );
            auto accL = bufLoc.get_access<cl::sycl::access::mode::read>( cgh );
            auto accOut = bufOut.get_access<cl::sycl::access::mode::write>( cgh );

            kernel_nd_item<3> kernel_3d( accG, accL, accOut );
            cgh.parallel_for( dataRange, kernel_3d );
        } );
    }
    /* check no errors are returned */
    if ( check_nd_item( dataOut.get(), globalSize[0], globalSize[1], globalSize[2] ) )
    {
        FAIL( log, "item API inconsistency" );
    }
}

/** test cl::sycl::nd_item
*/
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
    *  @param info, test_base::info structure as output
    */
    virtual void get_info( test_base::info& out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
    *  @param log, test transcript logging class
    */
    virtual void run( util::logger& log ) override
    {

        try
        {
            cts_selector selector;
            queue cmd_queue( selector );

            test_item( log, cmd_queue );

            cmd_queue.wait_and_throw();
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

} /* namespace test_nd_item__ */
