/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_destructors

namespace buffer_storage__
{
using namespace sycl_cts;
using namespace cl::sycl;
using namespace cl::sycl::access;

template <typename T>
class custom_alloc
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef size_t size_type;

    T * allocate( size_t n )
    {
        T * mem = static_cast<T*>( malloc( sizeof( T ) * n ) );
        if ( mem != nullptr )
            return mem;
        else
            throw std::bad_alloc();
    }

    void deallocate( T * p, size_t n )
    {
        free( p );
        return;
    }
};

template <typename alloc, typename T, int size, int dims>
class buffer_storage
{
public:
    void operator()( util::logger & log, range<dims> r )
    {
        unique_ptr<T> data     ( new T[size] );
        unique_ptr<T> data_uniq( new T[size] );
        shared_ptr<T> data_shrd( new T[size] );

        util::MUTEX m;

        memset( data.get(), 0, sizeof( T ) * size );
        memset( data_uniq.get(), 0, sizeof( T ) * size );
        memset( data_shrd.get(), 0, sizeof( T ) * size );

        {
            cl::sycl::buffer<T, dims, custom_alloc<T> > buf( data.get(), r );
            cl::sycl::buffer<T, dims, custom_alloc<T> > buf_uniq( std::move(data_uniq), r );
            cl::sycl::buffer<T, dims, custom_alloc<T> > buf_shrd( data_shrd, r );
        }
        {
            cl::sycl::buffer<T, dims, custom_alloc<T>> buf_shrd( data_shrd, r, &m );
            m.lock();
            memset( data_shrd.get(), 0xFF, size );
            m.unlock();
            weak_ptr<T> data_final;
            buf_shrd.set_final_data( data_final );
        }
    }
};

/**
* test cl::sycl::buffer initialization
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

    template <typename alloc, typename T>
    void test_buffers( util::logger & log )
    {
        const int size = 32;
        range<1> range1d( size );
        range<2> range2d( size, size );
        range<3> range3d( size, size, size );

        buffer_storage<alloc, T, size, 1> buf1d;
        buffer_storage<alloc, T, size * size, 2> buf2d;
        buffer_storage<alloc, T, size * size * size, 3> buf3d;

        buf1d( log, range1d );
        buf2d( log, range2d );
        buf3d( log, range3d );
    }

    /** execute the test
    */
    virtual void run( util::logger &log ) override
    {
        try
        {
            test_buffers<custom_alloc<int>,    int   >( log );
            test_buffers<custom_alloc<float>,  float >( log );
            test_buffers<custom_alloc<double>, double>( log );
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

} /* namespace buffer_constructors__ */
