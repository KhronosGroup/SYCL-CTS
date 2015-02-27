/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME explicit_pointer_constructors

namespace explicit_pointer_constructors__
{
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T>
class functor
{
public:
    template <int mode, int target>
    using acc1d = accessor<T, 1, mode, target>;

private:
    acc1d<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> m_ag;
    acc1d<cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer> m_ac;
    acc1d<cl::sycl::access::mode::read_write, cl::sycl::access::target::local> m_al;

public:
    functor(
        acc1d<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> a_g,
        acc1d<cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer> a_c,
        acc1d<cl::sycl::access::mode::read_write, cl::sycl::access::target::local> a_l )
        : m_ag( a_g )
        , m_ac( a_c )
        , m_al( a_l )
    {
        ;
    }

    void operator()()
    {
        T data[1];
        {
            cl::sycl::global_ptr<T>   gp( &ag[0] );
            cl::sycl::constant_ptr<T> cp( &ac[0] );
            cl::sycl::local_ptr<T>    lp( &al[0] );
            cl::sycl::private_ptr<T>  pp(  data  );

            gp[0] = 0;
            cp[0] = 0;
            lp[0] = 0;
            pp[0] = 0;
        }
        {
            cl::sycl::global_ptr<T>   gp( ag );
            cl::sycl::constant_ptr<T> cp( ac );
            cl::sycl::local_ptr<T>    lp( al );
        }
        {
            cl::sycl::global_ptr<T>   gp = (T *) ag;
            cl::sycl::constant_ptr<T> cp = (T *) ac;
            cl::sycl::local_ptr<T>    lp = (T *) al;
            cl::sycl::private_ptr<T>  pp = (T *) data;
        }
    }
};

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

    template <typename T>
    void test_type()
    {
        using namespace cl::sycl;
        using namespace cl::sycl::access;

        const int size = 64;
        cl::sycl::range<1> r( size );

        util::UNIQUE_PTR<T[]> data( new T[size] );

        cl::sycl::buffer<T, 1> buf( data.get(), r );

        cts_selector sel;
        cl::sycl::queue q( sel );
        q.submit( [&]( cl::sycl::handler & cgh )
        {
            auto acc_g = buf.template get_access<read_write>( cgh );
            auto acc_c = buf.template get_access<read, constant_buffer>( cgh );
            typename functor<T>::template acc1d<read_write, local> acc_l( r, cgh );

            functor<T> f( acc_g, acc_c, acc_l );

            cgh.single_task( f );
        } );

        q.wait_and_throw();

    }

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            test_type<int>();
            test_type<long long>();
            test_type<float>();
            test_type<double>();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace explicit_pointer_constructors__ */
