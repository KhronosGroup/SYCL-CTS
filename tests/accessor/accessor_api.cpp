/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME accessor_api

namespace accessor_api__
{
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T, int dims, int mode, int target>
class accessor_kernel
{
    accessor<T, dims, mode, target> acc;

public:

    accessor_kernel(accessor<T, dims, mode, target> a)
        : acc(a)
    {
        ;
    }

    void operator()( item<dims> i )
    {
        T var = acc[i.get_global_linear()];
    }
};

// TODO: When mode and target are enum classes, update this
// to use them exclusively, to disallow illegal values!
/**
* Class that constructs all the different kinds of accessor
* exclusively from buffers. Size depends on dims.
*/
template <typename T, int dims, int size, int mode, int target>
class accessors_buffers
{
public:
    void operator()( util::logger &log, queue &q, range<dims> &r )
    {
        T data[size];
        // Use memset because T might be some complicated type,
        // which means we can't use data = { 0 };
        memset( data, 0, sizeof( data ) );

        buffer<T, dims> buf( data, r );

        cl::sycl::command_group( q, [&]()
        {
            auto acc = buf.template get_access<mode, target>();

            accessor_kernel<T, dims, mode, target> kern( acc );
        } );
    }
};

template <typename T, int dims, int mode, int target>
class my_functor
{
    typedef accessor<T, dims, access::write, access::global_buffer> write_t;
    typedef accessor<T, dims, access::read , access::global_buffer> read_t;
    accessor<T, dims, mode, target> m_lcl;
    write_t m_out;
    read_t m_in;
public:
    my_functor( accessor<T, dims, mode, target> lcl, write_t out, read_t in )
        : m_lcl( lcl )
        , m_in ( in  )
        , m_out( out )
    { }

    void operator()( item<dims> it )
    {
        m_lcl[0] = m_in [0];
        m_out[0] = m_lcl[0];
    }
};

/**
* Class that is used only for the local accessor - local
* accessors can have no interaction with the host.
*/
template <typename T, int dims, int size, int mode, int target>
class accessors_local
{
public:
    void operator()( util::logger &log, queue &q, range<dims> &r, T expected, T init )
    {
        T input = init;
        T output = init;
        {
            cl::sycl::buffer<T, dims> buffer_input ( &input,  r / r);
            cl::sycl::buffer<T, dims> buffer_output( &output, r / r);

            cl::sycl::command_group( q, [&]()
            {
                accessor<T, dims, mode, target> acc(r);
                auto out = buffer_output.template  get_access<access::write, access::global_buffer >();
                auto in = buffer_input.template  get_access<access::read, access::global_buffer >();

                if( acc.get_size() != static_cast<size_t>( size * sizeof(T) ) )
                {
                    FAIL(log, "get_size() value is not as expected");
                }

                auto my_kernel = my_functor<T, dims, mode, target>( acc,out, in );
                parallel_for( nd_range<dims>( r, r / r ), my_kernel );
            } );
        }

        if ( output != expected )
        {
            FAIL(log, "Output value is not as expected");
        }

    }
};

template <typename T, int dims, int size, int mode>
class accessors_targets
{
public:
    void operator()( util::logger &log, queue &q, range<dims> &r, T expected, T init )
    {
        // Generate classes for each access target
        accessors_buffers<T, dims, size, mode, access::global_buffer> g;
        g( log, q, r );
        accessors_buffers<T, dims, size, mode, access::constant_buffer> c;
        c( log, q, r );
        accessors_buffers<T, dims, size, mode, access::host_buffer> hb;
        hb( log, q, r );
        accessors_local<T, dims, size, mode, access::local> l;
        l( log, q, r, expected, init );
    }
};

template <typename T, int dims, int size>
class accessors_modes
{
public:
    void operator()( util::logger &log, queue &q, range<dims> &r, T expected, T init )
    {
        // Generate classes for each access mode
        accessors_targets<T, dims, size, access::read> read;
        read( log, q, r, expected, init );
        accessors_targets<T, dims, size, access::write> write;
        write( log, q, r, expected, init );
        accessors_targets<T, dims, size, access::read_write> rw;
        rw( log, q, r, expected, init );
        accessors_targets<T, dims, size, access::discard_write> discard_write;
        discard_write(log, q, r, expected, init);
        accessors_targets<T, dims, size, access::discard_read_write> discard_rw;
        discard_rw(log, q, r, expected, init);
    }
};

/** test cl::sycl::image initialization
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
    void test_accessors( util::logger &log, cl::sycl::queue &q, T expeced, T init )
    {
        // Ranges of each dimension
        const int size = 32;
        range<1> range1d( size );
        range<2> range2d( size, size );
        range<3> range3d( size, size, size );

        accessors_modes<T, 1, size> acc1d;
        acc1d( log, q, range1d, expeced, init );
        accessors_modes<T, 2, size * size> acc2d;
        acc2d( log, q, range2d, expeced, init );
        accessors_modes<T, 3, size * size * size> acc3d;
        acc3d( log, q, range3d, expeced, init );
    }

    /** execute the test
    *  @param log, test transcript logging class
    */
    virtual void run( util::logger &log )
    {
        try
        {
            cts_selector selector;
            cl::sycl::queue queue( selector );

            test_accessors<int>( log, queue, 42, 0 );
            test_accessors<float>( log, queue, 42.0f, 0.0f );
            test_accessors<double>( log, queue, 42.0, 0.0 );
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
