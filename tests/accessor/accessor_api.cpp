/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME accessor_api

namespace accessor_api__
{
using namespace sycl_cts;

template <int dims>
util::VECTOR<cl::sycl::id<dims>> ids_list( cl::sycl::range<dims> &r );

template <>
util::VECTOR<cl::sycl::id<1>> ids_list<1>( cl::sycl::range<1> &r )
{
    util::VECTOR<cl::sycl::id<1>> ret;
    for ( size_t i = 0; i < r[0]; ++i )
        ret.push_back( cl::sycl::id<1>( i ) );
    return ret;
}

template <>
util::VECTOR<cl::sycl::id<2>> ids_list<2>( cl::sycl::range<2> &r )
{
    util::VECTOR<cl::sycl::id<2>> ret;
    for ( size_t i = 0; i < r[0]; ++i )
        for ( size_t j = 0; j < r[1]; ++j )
            ret.push_back( cl::sycl::id<2>( i, j ) );
    return ret;
}

template <>
util::VECTOR<cl::sycl::id<3>> ids_list<3>( cl::sycl::range<3> &r )
{
    util::VECTOR<cl::sycl::id<3>> ret;
    for ( size_t i = 0; i < r[0]; ++i )
        for ( size_t j = 0; j < r[1]; ++j )
            for ( size_t k = 0; k < r[2]; ++k )
                ret.push_back( cl::sycl::id<3>( i, j, k ) );
    return ret;
}

template <typename T, int dims, int size, int mode, int target>
class accessors_other_apis
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        q.submit( [&]( cl::sycl::handler &cgh )
        {
            cl::sycl::accessor<T, dims, mode, target> acc( r, cgh );

            /** check get_size() method
            */
            auto size = acc.get_size();

            if ( typeid( size ) != typeid( size_t ) )
            {
                FAIL( log, "accessor::get_size() does not return size_t" );
            }

            if ( size != size )
            {
                FAIL( log, "accessor is not the correct size" );
            }

            /** check get_event() method
            */
            auto accessorEvent = acc.get();

            if ( typeid( accessorEvent ) != typeid( cl::sycl::event ) )
            {
                FAIL( log, "accessor::get_event() does not return cl::sycl::event" );
            }
        } );
    }
};

template <typename T, int dims, int size, int mode, int target>
class accessors_writes;

template <typename T, int size, int mode, int target>
class accessors_writes<T, 1, size, mode, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<1> &r )
    {
        T data[size];
        memset( data, 0xFF, sizeof( data ) );

        cl::sycl::buffer<T, dims> buf( data, r );

        q.submit( [&]( cl::sycl::handler &cgh )
                  {
                      cl::sycl::accessor<T, dims, mode, target> acc( buf, cgh );

            cgh.parallel_for<class accessor_writes<T, dims, size, mode, target>>( r, [=]( cl::sycl::id<1> i )
            {
                auto ids = ids_list<dims>( r );

                size_t linearID = i[0];

                for ( auto i : ids )
                {
                    acc[i] = linearID;
                    acc[i[0]] = linearID;
                }
            } );
                  } );
    }
};

template <typename T, int size, int mode, int target>
class accessors_writes<T, 1, size, mode, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<2> &r )
    {
        T data[size];
        memset( data, 0xFF, sizeof( data ) );

        cl::sycl::buffer<T, dims> buf( data, r );

        q.submit( [&]( cl::sycl::handler &cgh )
                  {
                      cl::sycl::accessor<T, dims, mode, target> acc( buf, cgh );

            cgh.parallel_for<class accessor_writes<T, dims, size, mode, target>>( r, [=]( cl::sycl::id<2> i )
            {
                auto ids = ids_list<dims>( r );

                size_t linearID = i[0] + ( i[1] * r[0] );

                for ( auto i : ids )
                {
                    acc[i] = linearID;
                    acc[i[0]][i[1]] = linearID;
                }
            } );
                  } );
    }
};

template <typename T, int size, int mode, int target>
class accessors_writes<T, 1, size, mode, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<3> &r )
    {
        T data[size];
        memset( data, 0xFF, sizeof( data ) );

        cl::sycl::buffer<T, dims> buf( data, r );

        q.submit( [&]( cl::sycl::handler &cgh )
                  {
                      cl::sycl::accessor<T, dims, mode, target> acc( buf, cgh );

            cgh.parallel_for<class accessor_writes<T, dims, size, mode, target>>( r, [=]( cl::sycl::id<3> i )
            {
                auto ids = ids_list<dims>( r );

                size_t linearID = i[0] + ( i[1] * r[0] ) + ( i[2] * r[0] * r[1] );

                for ( auto i : ids )
                {
                    acc[i] = linearID;
                    acc[i[0]][i[1]][i[2]] = linearID;
                }
            } );
                  } );
    }
};

template <typename T, int dims, int size, int mode, int target>
class accessors_reads;

template <typename T, int size, int mode, int target>
class accessors_reads<T, 1, size, mode, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<1> &r )
    {
        T data[size];
        memset( data, 0xFF, sizeof( data ) );

        cl::sycl::buffer<T, dims> buf( data, r );

        q.submit( [&]( cl::sycl::handler &cgh )
                  {
                      cl::sycl::accessor<T, dims, mode, target> acc( buf, cgh );

            cgh.parallel_for<class accessor_reads<T, dims, size, mode, target>>( r, [=]( cl::sycl::id<1> i )
            {
                T elem;

                auto ids = ids_list<dims>( r );

                for ( auto i : ids )
                {
                    elem = acc[i];
                    elem = acc[i[0]];
                }
            } );
                  } );
    }
};

template <typename T, int size, int mode, int target>
class accessors_reads<T, 1, size, mode, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<2> &r )
    {
        T data[size];
        memset( data, 0xFF, sizeof( data ) );

        cl::sycl::buffer<T, dims> buf( data, r );

        q.submit( [&]( cl::sycl::handler &cgh )
                  {
                      cl::sycl::accessor<T, dims, mode, target> acc( buf, cgh );

            cgh.parallel_for<class accessor_reads<T, dims, size, mode, target>>( r, [=]( cl::sycl::id<2> i )
            {
                T elem;

                auto ids = ids_list<dims>( r );

                for ( auto i : ids )
                {
                    elem = acc[i];
                    elem = acc[i[0]][i[1]];
                }
            } );
                  } );
    }
};

template <typename T, int size, int mode, int target>
class accessors_reads<T, 1, size, mode, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<3> &r )
    {
        T data[size];
        memset( data, 0xFF, sizeof( data ) );

        cl::sycl::buffer<T, dims> buf( data, r );

        q.submit( [&]( cl::sycl::handler &cgh )
                  {
                      cl::sycl::accessor<T, dims, mode, target> acc( buf, cgh );

            cgh.parallel_for<class accessor_reads<T, dims, size, mode, target>>( r, [=]( cl::sycl::id<3> i )
            {
                T elem;

                auto ids = ids_list<dims>( r );

                for ( auto i : ids )
                {
                    elem = acc[i];
                    elem = acc[i[0]][i[1]][i[2]];
                }
            } );
                  } );
    }
};

template <typename T, int dims, int size, int mode, int target>
class accessors_subscripts;

template <typename T, int dims, int size, int target>
class accessors_subscripts<T, dims, size, cl::sycl::access::mode::read, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_reads<T, dims, size, cl::sycl::access::mode::read, target> read;
        read( log, q, r );
    }
};

template <typename T, int dims, int size, int target>
class accessors_subscripts<T, dims, size, cl::sycl::access::mode::write, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_writes<T, dims, size, cl::sycl::access::mode::write, target> write;
        write( log, q, r );
    }
};

template <typename T, int dims, int size, int target>
class accessors_subscripts<T, dims, size, cl::sycl::access::mode::read_write, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_reads<T, dims, size, cl::sycl::access::mode::read_write, target> read;
        read( log, q, r );
        accessors_writes<T, dims, size, cl::sycl::access::mode::read_write, target> write;
        write( log, q, r );
    }
};

template <typename T, int dims, int size, int target>
class accessors_subscripts<T, dims, size, cl::sycl::access::mode::discard_write, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_writes<T, dims, size, cl::sycl::access::mode::discard_write, target> write;
        write( log, q, r );
    }
};

template <typename T, int dims, int size, int target>
class accessors_subscripts<T, dims, size, cl::sycl::access::mode::discard_read_write, target>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_reads<T, dims, size, cl::sycl::access::mode::discard_read_write, target> read;
        read( log, q, r );
        accessors_writes<T, dims, size, cl::sycl::access::mode::discard_read_write, target> write;
        write( log, q, r );
    }
};

template <typename T, int dims, int size, int mode, int target>
class accessors_apis
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_subscripts<T, dims, size, mode, target> subscripts;
        subscripts( log, q, r );
        accessors_other_apis<T, dims, size, mode, target> otherAPIs;
        otherAPIs( log, q, r );
    }
};

template <typename T, int dims, int size, int target>
class accessors_modes;

template <typename T, int dims, int size>
class accessors_modes<T, dims, size, cl::sycl::access::target::global_buffer>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_apis<T, dims, size, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> read;
        read( log, q, r );
        accessors_apis<T, dims, size, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> write;
        write( log, q, r );
        accessors_apis<T, dims, size, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> rw;
        rw( log, q, r );
        accessors_apis<T, dims, size, cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer> discard_write;
        discard_write( log, q, r );
        accessors_apis<T, dims, size, cl::sycl::access::mode::discard_read_write, cl::sycl::access::target::global_buffer> discard_rw;
        discard_rw( log, q, r );
    }
};

template <typename T, int dims, int size>
class accessors_modes<T, dims, size, cl::sycl::access::target::host_buffer>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_apis<T, dims, size, cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer> read;
        read( log, q, r );
        accessors_apis<T, dims, size, cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer> write;
        write( log, q, r );
        accessors_apis<T, dims, size, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> rw;
        rw( log, q, r );
        accessors_apis<T, dims, size, cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer> discard_write;
        discard_write( log, q, r );
    }
};

template <typename T, int dims, int size>
class accessors_modes<T, dims, size, cl::sycl::access::target::constant_buffer>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_apis<T, dims, size, access::mode::read, cl::sycl::access::target::constant_buffer> read;
        read( log, q, r );
    }
};

template <typename T, int dims, int size>
class accessors_modes<T, dims, size, cl::sycl::access::target::local>
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        accessors_apis<T, dims, size, access::mode::read_write, cl::sycl::access::target::local> rw;
        rw( log, q, r );
    }
};

template <typename T, int dims, int size>
class accessors_targets
{
public:
    void operator()( util::logger &log, cl::sycl::queue &q, cl::sycl::range<dims> &r )
    {
        // Generate classes for each access target
        accessors_modes<T, dims, mode, access::target::global_buffer> g;
        g( log, q, r );
        accessors_modes<T, dims, mode, access::target::constant_buffer> c;
        c( log, q, r );
        accessors_modes<T, dims, mode, access::target::host_buffer> h;
        h( log, q, r );
        accessors_modes<T, dims, mode, access::target::local> l;
        l( log, q, r );
    }
};

/** test cl::sycl::image initialization
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

    template <typename T>
    void test_accessors( util::logger &log, cl::sycl::queue &q )
    {
        // Ranges of each dimension
        const int size = 32;
        cl::sycl::range<1> range1d( size );
        cl::sycl::range<2> range2d( size, size );
        cl::sycl::range<3> range3d( size, size, size );

        accessors_targets<T, 1, size> acc1d;
        acc1d( log, q, range1d );
        accessors_targets<T, 2, size * size> acc2d;
        acc2d( log, q, range2d );
        accessors_targets<T, 3, size * size * size> acc3d;
        acc3d( log, q, range3d );
    }

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cts_selector selector;
            cl::sycl::queue queue( selector );

            test_accessors<int>( log, queue );
            test_accessors<float>( log, queue );
            test_accessors<double>( log, queue );

            queue.wait_and_throw();
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

} /* namespace accessor_api__ */
