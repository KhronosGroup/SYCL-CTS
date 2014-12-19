/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME id_api

namespace id_api__
{
using namespace sycl_cts;

template<int dims>
class test_kernel
{

};

//golden values
#define X 16
#define Y 32
#define Z 64
#define LOCAL 2

template<int dims>
class test_id
{
public:
    void operator()(util::logger& log, cl::sycl::range<dims> global,
                    cl::sycl::range<dims> local, cl::sycl::queue myQueue )
    {
        // for testing get()
        int error;

        for (int i = 0; i < dims; i++)
        {
            error = 0; //ney error
        }

        {
            cl::sycl::buffer<int, 1> errorBuffer(&error, 1);

            cl::sycl::command_group(myQueue, [&]() {
                auto myRange = cl::sycl::nd_range<dims>(global,
                        local);

                auto errorPtr = errorBuffer.get_access<cl::sycl::access::read_write>();

                auto myKernel = ([=](cl::sycl::item<dims> item) {
                   cl::sycl::id<dims> id(item);
                   //create check table
                   int check[] = {X,Y,Z};

                   if(id.get_global_linear() > X * Y * Z || id.get_global_linear() < 0)
                   {
                       errorPtr[0] = __LINE__;
                   }

                   if(id.get_local_linear() > LOCAL * LOCAL * LOCAL || id.get_local_linear() < 0)
                   {
                       errorPtr[0] = __LINE__;
                   }

                   for(int i =0; i < dims; i++)
                   {
                       if(id.get(i) > check[i] || id[i] > check[i])
                       {
                            //report error
                            errorPtr[0] = i ;
                       }
                    }

                   for(int i =0; i < dims; i++)
                    {
                       cl::sycl::id<dims> idTwo(id);

                        //operators
                        //*
                        id = id * idTwo;
                        for(int k = 0; k < dims; k++)
                        {
                            if(id.get(k) != idTwo.get(k) * idTwo.get(k))
                            {
                                errorPtr[0] = __LINE__;
                            }
                        }

                        // /
                        bool safeDev = true;
                        for(int j = 0; j < dims; j++)
                        {
                          if(idTwo.get(j) == 0)
                              safeDev = false;
                        }

                        if(safeDev)
                        {
                            id = id / idTwo;

                            for(int k = 0; k < dims; k++)
                            {
                                if(id.get(k) != idTwo.get(k) )
                                {
                                    errorPtr[0] = __LINE__;
                                }
                            }
                        } else {
                            id = idTwo;
                        }

                        //reset - otherwise dims 1 fails on ==
                        //possibly optimisation error or I am missing something.
                        id = idTwo;

                        //+
                        id = id + idTwo;
                        for(int k = 0; k < dims; k++)
                        {
                            if(id.get(k) < idTwo.get(k) + idTwo.get(k) )
                            {
                                errorPtr[0] = __LINE__;
                            }
                        }

                        //-
                        id = id - idTwo;
                        for(int k = 0; k < dims; k++)
                        {
                            if(id.get(k) != idTwo.get(k) )
                            {
                                errorPtr[0] = __LINE__;
                            }
                        }

                        // ==
                        if(id == idTwo)
                        {
                            errorPtr[0] = 0;
                        }
                        else
                        {
                            errorPtr[0] = __LINE__;
                        }
                   }

                });
                cl::sycl::parallel_for<class test_kernel<dims>>(myRange, myKernel);
            });
        }

        CHECK_VALUE(log, error, 0, dims );

    }
};

/** test cl::sycl::range::get(int index) return size_t
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
            //use accross all the dimentions
            cl::sycl::default_selector selector;
            cl::sycl::queue myQueue(selector);
            //templated approach
            {
                cl::sycl::range<1> range1dG( X );
                cl::sycl::range<2> range2dG( X, Y );
                cl::sycl::range<3> range3dG( X, Y, Z );

                cl::sycl::range<1> range1dL( LOCAL );
                cl::sycl::range<2> range2dL( LOCAL, LOCAL );
                cl::sycl::range<3> range3dL( LOCAL, LOCAL, LOCAL );

                test_id<1> test1d;
                test1d( log, range1dG, range1dL, myQueue );
                test_id<2> test2d;
                test2d( log, range2dG, range2dL, myQueue );
                test_id<3> test3d;
                test3d( log, range3dG, range3dL, myQueue );

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
