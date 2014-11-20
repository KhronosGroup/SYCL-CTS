/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME buffer_api

namespace sycl_cts
{

/** Generic buffer API test function
 *
 * TODO: add support for dimension > 1
 */
template <typename T, int SIZE, int DIM>
bool testBuffer( util::logger &log )
{
    try
    {
        /* stack allocate a host buffer */
        T hostBuffer[SIZE];
        memset( hostBuffer, 0, sizeof( T ) * SIZE );

        /* create a range for this buffer */
        cl::sycl::range<DIM> syclRange( SIZE );

        /* create a sycl buffer from the host buffer */
        /* TODO: update construction to use range */
        cl::sycl::buffer<T, DIM> syclBuffer( hostBuffer, SIZE );

        /* check the buffer returns the correct range */
        auto ret_range = syclBuffer.get_range();
        if ( typeid( ret_range ) != typeid(cl::sycl::range<DIM>))
        {
            FAIL( log,
                  "cl::sycl::buffer::get_range does not return "
                  "cl::sycl::range!" );
            return false;
        }

        /* check the buffer returns the correct element count */
        auto syclCount = syclBuffer.get_count();
        if ( typeid( syclCount ) != typeid( size_t ) )
        {
            FAIL( log, "buffer.get_count() does not return size_t" );
            return false;
        }
        if ( syclCount != SIZE )
        {
            FAIL( log,
                  "cl::sycl::buffer::get_count does not return the correct "
                  "number of elements" );
            return false;
        }

        /* check the buffer returns the correct byte size */
        auto syclSize = syclBuffer.get_size();
        if ( typeid( syclSize ) != typeid( size_t ) )
        {
            FAIL( log, "buffer.get_size() does not return size_t" );
            return false;
        }
        if ( syclSize != SIZE * sizeof( T ) )
        {
            FAIL( log,
                  "cl::sycl::buffer::get_size does not return the correct size "
                  "of the buffer" );
            return false;
        }

        /* retrieve an accessor to the buffer */
        auto syclAccess = syclBuffer.template get_access<cl::sycl::access::read_write>();
    }
    catch ( cl::sycl::sycl_error e )
    {
        log_exception( log, e );
        FAIL( log, "sycl exception caught" );
        return false;
    }

    return true;
}

/** test cl::sycl::buffer api
 */
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
    *  @param info, test_base::info structure as output
    */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
    *  @param log, test transcript logging class
    */
    virtual void run( util::logger &log )
    {
        const int defSize = 64;

        /* test signed types */
        testBuffer<int8_t, defSize, 1>( log );
        testBuffer<int16_t, defSize, 1>( log );
        testBuffer<int32_t, defSize, 1>( log );
        testBuffer<int64_t, defSize, 1>( log );

        /* test unsigned types */
        testBuffer<uint8_t, defSize, 1>( log );
        testBuffer<uint16_t, defSize, 1>( log );
        testBuffer<uint32_t, defSize, 1>( log );
        testBuffer<uint64_t, defSize, 1>( log );

        /* test float types */
        testBuffer<float, defSize, 1>( log );
        testBuffer<double, defSize, 1>( log );

        /* test vector types */
        testBuffer<cl::sycl::float2, defSize, 1>( log );
        testBuffer<cl::sycl::float3, defSize, 1>( log );
        testBuffer<cl::sycl::float4, defSize, 1>( log );
        testBuffer<cl::sycl::float8, defSize, 1>( log );
        testBuffer<cl::sycl::float16, defSize, 1>( log );
    }
};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

};  // sycl_cts
