/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME buffer_constructors

namespace sycl_cts
{

// TODO: fully implement this class
template <typename T>
struct cts_storage : cl::sycl::storage
{
    virtual void in_use()
    {
    }
    virtual void completed()
    {
    }
};

template <int size, typename T, int dims>
class call_all_ctors;

template <int size, typename T>
class call_all_ctors<size, T, 1>
{
    typedef cl::sycl::buffer<T, 1> buf1d;

public:
    call_all_ctors( cl::sycl::range<1> r )
    {
        /* allocate host side buffer */
        T data[size];
        memset( data, 0, sizeof( data ) );

        buf1d buf_sized( size );
        buf1d buf_copy( buf_sized );

        buf1d buf_const( (const T *)data, size );
        cts_storage<T> store;
        buf1d buf( r );
        buf1d buf_range( data, r );
        buf1d buf_range_sized( data, size );
        buf1d buf_store( store, r );
        buf1d buf_iter( std::begin( data ), std::end( data ) );
        cl::sycl::index index;  // not impl
        cl::sycl::range<1> sub_range( 1 );
        buf1d buf_sub( buf_sized, index, sub_range );

    }
};

template <int size, typename T>
class call_all_ctors<size, T, 2>
{
    typedef cl::sycl::buffer<T, 2> buf2d;

public:
    call_all_ctors( cl::sycl::range<2> r )
    {
        /* allocate host side buffer */
        T data[size * size];
        memset( data, 0, sizeof( data ) );

        buf2d buf_sized( size, size );
        buf2d buf_copy( buf_sized );

        buf2d buf_const( (const T *)data, size, size );
        cts_storage<T> store;
        buf2d buf( r );
        buf2d buf_range( data, r );
        buf2d buf_range_sized( data, size, size );
        buf2d buf_store( store, r );
        buf2d buf_iter( std::begin( data ), std::end( data ) );
        cl::sycl::index index;  // not impl
        cl::sycl::range<2> sub_range( 1 );
        buf2d buf_sub( buf_sized, index, sub_range );

    }
};

template <int size, typename T>
class call_all_ctors<size, T, 3>
{
    typedef cl::sycl::buffer<T, 3> buf3d;

public:
    call_all_ctors( cl::sycl::range<3> r )
    {
        /* allocate host side buffer */
        T data[size * size * size];
        memset( data, 0, sizeof( data ) );

        buf3d buf_sized( size, size, size );
        buf3d buf_copy( buf_sized );
        buf3d buf_range_sized( data, size, size, size );

        buf3d buf_const( (const T *)data, size, size, size );
        cts_storage<T> store;
        buf3d buf( r );
        buf3d buf_range( data, r );
        buf3d buf_store( store, r );
        buf3d buf_iter( std::begin( data ), std::end( data ) );
        cl::sycl::index index;  // not impl
        cl::sycl::range<3> sub_range( 1 );
        buf3d buf_sub( buf_sized, index, sub_range );

    }
};

template <int size, typename T>
class buffer_ctors_dims
{
public:
    buffer_ctors_dims()
    {
        cl::sycl::range<1> range_1( size );
        cl::sycl::range<2> range_2( size, size );
        cl::sycl::range<3> range_3( size, size, size );

        call_all_ctors<size, T, 1> test_1( range_1 );
        call_all_ctors<size, T, 2> test_2( range_2 );
        call_all_ctors<size, T, 3> test_3( range_3 );
    }
};

template <int size>
class buffer_ctors_types
{
public:
    buffer_ctors_types()
    {
        /* test signed scalars */
        buffer_ctors_dims<size, int8_t> int8_buffers;
        buffer_ctors_dims<size, int16_t> int16_buffers;
        buffer_ctors_dims<size, int32_t> int32_buffers;
        buffer_ctors_dims<size, int64_t> int64_buffers;

        /* test unsigned scalars */
        buffer_ctors_dims<size, uint8_t> uint8_buffers;
        buffer_ctors_dims<size, uint16_t> uint16_buffers;
        buffer_ctors_dims<size, uint32_t> uint32_buffers;
        buffer_ctors_dims<size, uint64_t> uint64_buffers;

        /* test floating point types */
        buffer_ctors_dims<size, float> bufFloat;
        buffer_ctors_dims<size, double> bufDouble;

        /* test vector types */
        buffer_ctors_dims<size, cl::sycl::float2> bufVec2;
        buffer_ctors_dims<size, cl::sycl::float3> bufVec3;
        buffer_ctors_dims<size, cl::sycl::float4> bufVec4;
        buffer_ctors_dims<size, cl::sycl::float8> bufVec8;
        buffer_ctors_dims<size, cl::sycl::float16> bufVec16;
    }
};

/** test cl::sycl::buffer initialization
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
        try
        {
            const int size = 32;
            buffer_ctors_types<size> test;
        }
        catch ( cl::sycl::sycl_error e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
static util::test_proxy<TEST_NAME> proxy;

};  // sycl_cts
