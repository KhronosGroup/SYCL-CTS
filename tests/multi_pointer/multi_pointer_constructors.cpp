/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME multi_pointer_constructors

namespace TEST_NAMESPACE
{
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T>
class kernel_name;

/** 
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

    template <typename T, int size>
    void test_type( cl::sycl::queue & q )
    {
        using namespace cl::sycl::access;
        using namespace cl::sycl::address_space;

        util::UNIQUE_PTR<T[]> data_host( new T[size] );
        memset( data_host.get(), 0, sizeof( T ) * size );

        cl::sycl::range<1> r( size );
        buffer<T, 1>buf( data_host.get(), r );
        q.submit( [&]( cl::sycl::handler & cgh )
        {
            accessor<T, 1, read_write, global_buffer> acc_g( buf, cgh );
            accessor<T, 1, read, constant_buffer> acc_c( buf, cgh );
            accessor<T, 1, read_write, local> acc_l( r, cgh );

            cgh.single_task<kernel_name<T>>( [=]()
            {
                T data[size];
                {
                    auto ptr_g = multi_ptr::make_ptr<T, global_space>( &acc_g[0] );
                    auto ptr_l = multi_ptr::make_ptr<T, local_space>( &acc_l[0] );
                    auto ptr_c = multi_ptr::make_ptr<T, constant_space>( &acc_c[0] );
                    auto ptr_p = multi_ptr::make_ptr<T, private_space>( data );

                    *ptr_g = 0;
                    *ptr_l = 0;
                    *ptr_c = 0;
                    *ptr_p = 0;

                    for( int i = 1; i < size; ++i)
                    {
                        ptr[i] = 0;
                        ptr[i] = 0;
                        ptr[i] = 0;
                        ptr[i] = 0;
                    }

                    {
                        global_ptr<T>   explicit_g_ptr = (global_ptr<T>) ptr_g;
                        local_ptr<T>    explicit_l_ptr = (local_ptr<T>) ptr_l;
                        constant_ptr<T> explicit_c_ptr = (constant_ptr<T>) ptr_c;
                        private_ptr<T>  explicit_p_ptr = (private_ptr<T>) ptr_p;
                    }
                    {
                        global_ptr<T>   explicit_g_ptr = ptr_g.pointer();
                        local_ptr<T>    explicit_l_ptr = ptr_l.pointer();
                        constant_ptr<T> explicit_c_ptr = ptr_c.pointer();
                        private_ptr<T>  explicit_p_ptr = ptr_p.pointer();
                    }
                    {
                        T * raw_g_ptr = (T*) ptr_g;
                        T * raw_l_ptr = (T*) ptr_l;
                        T * raw_c_ptr = (T*) ptr_c;
                        T * raw_p_ptr = (T*) ptr_p;
                    }
                }
                {
                    auto ptr_g = multi_ptr::make_ptr<T, global_space>( acc_g );
                    auto ptr_l = multi_ptr::make_ptr<T, local_space>( acc_l );
                    auto ptr_c = multi_ptr::make_ptr<T, constant_space>( acc_c );
                }
            } );
        } );
    }

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            const int size = 32;
            cl::sycl::default_selector sel;
            cl::sycl::queue  q( sel );
            test_type<int, size>( q );
            test_type<float, size>( q );

            q.wait_and_throw();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
