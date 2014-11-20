/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME accessor_constructors

using namespace cl::sycl;

namespace sycl_cts
{

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

        buffer<T, dims> buf( data, size );

        cl::sycl::command_group( q, [&]()
                                 {
            auto acc = buf.template get_access<mode, target>();

            accessor<T, dims, mode, target> acc_range( r );
            range<dims> base( r );
            range<dims> offset( r );
            for ( int i = 0; i < dims; ++i )
            {
                base[i] /= 4;
                offset[i] /= 2;
            }
            accessor<T, dims, mode, target> acc_sub( buf, base, offset );

        } );
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
    void operator()( util::logger &log, queue &q, range<dims> &r )
    {
        cl::sycl::command_group( q, [&]()
                                 {

            accessor<T, dims, mode, target> acc( r );

            accessor<T, dims, mode, target> acc_size( size );
        } );
    }
};

template <typename T, int dims, int size, int mode, int target>
class accessors_images
{
public:
    void operator()( util::logger &log, queue &q, range<dims> &r )
    {

        cl::sycl::image<dims> img;

    }
};

template <typename T, int dims, int size, int mode>
class accessors_targets
{
public:
    void operator()( util::logger &log, queue &q, range<dims> &r )
    {
        // Generate classes for each access target
        accessors_buffers<T, dims, size, mode, access::global_buffer> g;
        g( log, q, r );
        accessors_buffers<T, dims, size, mode, access::constant_buffer> c;
        c( log, q, r );
        accessors_buffers<T, dims, size, mode, access::host_buffer> hb;
        hb( log, q, r );
        accessors_local<T, dims, size, mode, access::local> l;
        l( log, q, r );
        accessors_images<T, dims, size, mode, access::image> i;
        i( log, q, r );
        accessors_images<T, dims, size, mode, access::host_image> hi;
        hi( log, q, r );

        accessors_images<T, mode, access::image_array> a;
        a( log, q, r );

    }
};

template <typename T, int dims, int size>
class accessors_modes
{
public:
    void operator()( util::logger &log, queue &q, range<dims> &r )
    {
        // Generate classes for each access mode
        accessors_targets<T, dims, size, access::read> read;
        read( log, q, r );
        accessors_targets<T, dims, size, access::write> write;
        write( log, q, r );
        accessors_targets<T, dims, size, access::read_write> rw;
        rw( log, q, r );

        accessors_targets<T, dims, size, access::discard_write> discard_write;
        discard_write( log, q, r );
        accessors_targets<T, dims, size, access::discard_read_write> discard_rw;
        discard_rw( log, q, r );

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
    void test_accessors( util::logger &log, cl::sycl::queue &q )
    {
        // Ranges of each dimension
        const int size = 32;
        range<1> range1d( size );
        range<2> range2d( size, size );
        range<3> range3d( size, size, size );

        accessors_modes<T, 1, size> acc1d;
        acc1d( log, q, range1d );

        accessors_modes<T, 2, size * size> acc2d;
        acc2d( log, q, range2d );
        accessors_modes<T, 3, size * size * size> acc3d;
        acc3d( log, q, range3d );

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

            test_accessors<int>( log, queue );
            test_accessors<float>( log, queue );
            test_accessors<cl::sycl::int2>( log, queue );
            test_accessors<double>( log, queue );
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
