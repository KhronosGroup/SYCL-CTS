/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME id_api

namespace id_api__
{
using namespace sycl_cts;

template<int dims>
class test_kernel{};

template<int dims>
class test_id
{
public:
    //golden values
    static const int m_x = 16;
    static const int m_y = 32;
    static const int m_z = 64;
    static const int m_local = 2;
    int m_error[20];

    void operator()( util::logger& log,
                     cl::sycl::range<dims> global,
                     cl::sycl::range<dims> local,
                     cl::sycl::queue q )
    {
        // for testing get()
        for (int i = 0; i < 20; i++)
        {
            m_error[i] = 0; //no error
        }

        {
            cl::sycl::buffer<int, 1> error_buffer(m_error, cl::sycl::range<1> ( 20 ) );

            q.submit( [&]( cl::sycl::handler& cgh )
            {
                auto my_range = cl::sycl::nd_range<dims>( global, local );

                auto error_ptr = error_buffer.get_access<cl::sycl::cl::sycl::access::mode::read_write>(cgh);

                auto my_kernel = ( [=](cl::sycl::item<dims> item)
                {
                   int m_itteration = 0;

                   cl::sycl::id<dims> id( item );
                   //create check table
                   int check[] = { m_x, m_y, m_z };

                    for(int i =0; i < dims; i++)
                    {
                       if(id.get(i) > check[i] || id[i] > check[i])
                       {
                            //report an error
                            error_ptr[m_itteration] = __LINE__ ;
                            m_itteration++;
                       }
                    }

                    {
                        cl::sycl::id<dims> id_two( id );

                        //operators
                        //*
                        id = id * id_two;
                        for(int k = 0; k < dims; k++)
                        {
                            if(id.get(k) != id_two.get(k) * id_two.get(k))
                            {
                                error_ptr[m_itteration] = __LINE__;
                                m_itteration++;
                            }
                        }

                        // /
                        bool is_zero = false;
                        for(int j = 0; j < dims; j++)
                        {
                          if(id_two.get(j) == 0)
                              is_zero = true;
                        }

                        if(!is_zero)
                        {
                            id = id / id_two;

                            for(int k = 0; k < dims; k++)
                            {
                                if(id.get(k) != id_two.get(k) )
                                {
                                    error_ptr[m_itteration] = __LINE__;
                                    m_itteration++;
                                }
                            }
                        }else {
                            id = id_two;
                        }

                        
                        /// can someone check that?
                        //reset - otherwise dims 1 fails on ==
                        //possibly optimisation error or I am missing something.
                        id = id_two;

                        //+
                        id = id + id_two;
                        for(int k = 0; k < dims; k++)
                        {
                            if(id.get(k) < id_two.get(k) + id_two.get(k) )
                            {
                                error_ptr[ m_itteration ] = __LINE__;
                                m_itteration++;
                            }
                        }

                        //-
                        id = id - id_two;
                        for(int k = 0; k < dims; k++)
                        {
                            if(id.get(k) != id_two.get(k) )
                            {
                                error_ptr[ m_itteration ] = __LINE__;
                                m_itteration++;
                            }
                        }

                        // ==
                        if(id == id_two)
                        {
                            error_ptr[ m_itteration ] = 0;
                             m_itteration++;
                        }
                        else
                        {
                            error_ptr[ m_itteration ] = __LINE__;
                            m_itteration++;
                        }
                    }
                });
                cgh.parallel_for<class test_kernel<dims>>( my_range, my_kernel );
            });

            q.wait_and_throw();

        }
        for(int i =0; i < 20; i++)
        {
            CHECK_VALUE(log, m_error[i], 0, i );
        }
    }
};

/** test cl::sycl::range::get(int index) return size_t
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

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            //use accross all the dimentions
            cl::sycl::default_selector selector;
            cl::sycl::queue my_queue(selector);
            //templated approach
            {
                cl::sycl::range<1> range_1d_g( test_id<1>::m_x );
                cl::sycl::range<2> range_2d_g( test_id<2>::m_x, test_id<2>::m_y );
                cl::sycl::range<3> range_3d_g( test_id<3>::m_x, test_id<3>::m_y, test_id<3>::m_z );

                cl::sycl::range<1> range_1d_l( test_id<1>::m_local );
                cl::sycl::range<2> range_2d_l( test_id<2>::m_local, test_id<2>::m_local );
                cl::sycl::range<3> range_3d_l( test_id<3>::m_local, test_id<3>::m_local, test_id<3>::m_local );

                test_id<1> test1d;
                test1d( log, range_1d_g, range_1d_l, my_queue );
                test_id<2> test2d;
                test2d( log, range_2d_g, range_2d_l, my_queue );
                test_id<3> test3d;
                test3d( log, range_3d_g, range_3d_l, my_queue );
            }
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

} /* namespace id_api__ */
