/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_api

namespace buffer_api__
{
using namespace sycl_cts;

/**
 * Generic buffer API test function
 **/
template <typename T, int size, int dims>
bool test_buffer( util::logger &log, cl::sycl::range<dims> & r )
{
    try
    {
        /* stack overflows when T is large */
        util::UNIQUE_PTR<T> data( new T[size] );
        memset( data.get(), 0, sizeof( T ) * size );

        /* create a sycl buffer from the host buffer */
        cl::sycl::buffer<T, dims> buf( data.get(), r );

        /* check the buffer returns the correct range */
        auto ret_range = buf.get_range();
        if ( typeid( ret_range ) != typeid(cl::sycl::range<dims>))
        {
            FAIL( log, "cl::sycl::buffer::get_range does not return "
                       "cl::sycl::range!" );
            return false;
        }

        /* check the buffer returns the correct element count */
        auto syclCount = buf.get_count();
        if ( typeid( syclCount ) != typeid( size_t ) )
        {
            FAIL( log, "buffer.get_count() does not return size_t" );
            return false;
        }

        if ( syclCount != size )
        {
            FAIL( log,
                  "cl::sycl::buffer::get_count does not return the correct "
                  "number of elements" );
            return false;
        }

        /* check the buffer returns the correct byte size */
        auto syclSize = buf.get_size();
        if ( typeid( syclSize ) != typeid( size_t ) )
        {
            FAIL( log, "buffer.get_size() does not return size_t" );
            return false;
        }
        if ( syclSize != size * sizeof( T ) )
        {
            FAIL( log,
                  "cl::sycl::buffer::get_size does not return the correct size"
                  " of the buffer" );
            return false;
        }

        cl::sycl::queue q;
        cl::sycl::command_group( q, [&]()
        {
            auto syclAccess =
                buf.template get_access<cl::sycl::access::read_write>();
        } );
    }
    catch ( cl::sycl::exception e )
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

    template <typename T>
    void test_type( util::logger & log )
    {
        const int size = 64;
        cl::sycl::range<1> range1d(size            );
        cl::sycl::range<2> range2d(size, size      );
        cl::sycl::range<3> range3d(size, size, size);

        test_buffer<T, size, 1>( log, range1d );
        test_buffer<T, size, 2>( log, range2d );
        test_buffer<T, size, 3>( log, range3d );
    }

    /** execute the test
    *  @param log, test transcript logging class
    */
    virtual void run( util::logger &log )
    {

        /* test signed types */
        test_type<int8_t >( log );
        test_type<int16_t>( log );
        test_type<int32_t>( log );
        test_type<int64_t>( log );

        /* test unsigned types */
        test_type<uint8_t >( log );
        test_type<uint16_t>( log );
        test_type<uint32_t>( log );
        test_type<uint64_t>( log );

        /* test float types */
        test_type<float >( log );
        test_type<double>( log );

        /* test vector types */
        test_type<cl::sycl::float3 >( log );
        test_type<cl::sycl::float2 >( log );
        test_type<cl::sycl::float4 >( log );
        test_type<cl::sycl::float8 >( log );
        test_type<cl::sycl::float16>( log );

        test_type<cl::sycl::double2 >( log );
        test_type<cl::sycl::double3 >( log );
        test_type<cl::sycl::double4 >( log );
        test_type<cl::sycl::double8 >( log );
        test_type<cl::sycl::double16>( log );

        test_type<cl::sycl::char2 >( log );
        test_type<cl::sycl::char3 >( log );
        test_type<cl::sycl::char4 >( log );
        test_type<cl::sycl::char8 >( log );
        test_type<cl::sycl::char16>( log );

        test_type<cl::sycl::int2 >( log );
        test_type<cl::sycl::int3 >( log );
        test_type<cl::sycl::int4 >( log );
        test_type<cl::sycl::int8 >( log );
        test_type<cl::sycl::int16>( log );

        test_type<cl::sycl::short2 >( log );
        test_type<cl::sycl::short3 >( log );
        test_type<cl::sycl::short4 >( log );
        test_type<cl::sycl::short8 >( log );
        test_type<cl::sycl::short16>( log );

        test_type<cl::sycl::long2 >( log );
        test_type<cl::sycl::long3 >( log );
        test_type<cl::sycl::long4 >( log );
        test_type<cl::sycl::long8 >( log );
        test_type<cl::sycl::long16>( log );

        test_type<cl::sycl::uchar2 >( log );
        test_type<cl::sycl::uchar3 >( log );
        test_type<cl::sycl::uchar4 >( log );
        test_type<cl::sycl::uchar8 >( log );
        test_type<cl::sycl::uchar16>( log );

        test_type<cl::sycl::uint2 >( log );
        test_type<cl::sycl::uint3 >( log );
        test_type<cl::sycl::uint4 >( log );
        test_type<cl::sycl::uint8 >( log );
        test_type<cl::sycl::uint16>( log );

        test_type<cl::sycl::ushort2 >( log );
        test_type<cl::sycl::ushort3 >( log );
        test_type<cl::sycl::ushort4 >( log );
        test_type<cl::sycl::ushort8 >( log );
        test_type<cl::sycl::ushort16>( log );

        test_type<cl::sycl::ulong2 >( log );
        test_type<cl::sycl::ulong3 >( log );
        test_type<cl::sycl::ulong4 >( log );
        test_type<cl::sycl::ulong8 >( log );
        test_type<cl::sycl::ulong16>( log );
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace buffer_api__ */
