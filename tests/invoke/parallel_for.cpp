/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME single_parallel_for

namespace parallel_for__
{
using namespace sycl_cts;

#define MEANING 42.0f
#define SIZE 16

/** test cl::sycl::kernel from functor
 */
class TEST_NAME : public sycl_cts::util::test_base
{
public:

    /** return information about this test
     *  @param out, test_base::info structure as output
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
            using namespace cl::sycl;
            default_selector sel;
            queue queue(sel);

            //range,id
            {
                float f[SIZE];
                {
                    cl::sycl::buffer<float,1> buf(&f[0], cl::sycl::range<1>(SIZE));
                    command_group(queue, [&] () {
                        auto a_dev = buf.get_access<access::read_write>();

                        parallel_for<class BASED_ON_ID>(
                                    range<1>(SIZE),
                                    [=] (id<1> index)
                        {
                            a_dev[index] = MEANING;
                        });
                    });
                }

                for(int i = 0; i < SIZE; i++)
                {
                    if(f[i] != MEANING)
                    {
                        CHECK_VALUE( log, f[i], MEANING, i );
                    }
                }
            }

            //range,item
            {
                float f[SIZE];
                {
                    cl::sycl::buffer<float,1> buf(&f[0], cl::sycl::range<1>(SIZE));
                    command_group(queue, [&] () {
                        auto a_dev = buf.get_access<access::read_write>();

                        parallel_for<class BASED_ON_ITEM>(
                                    range<1>(range<1>(SIZE)),
                                    [=](item<1> itm)
                        {
                            size_t index = itm.get_global();
                            a_dev[index] = MEANING;
                        });
                    });
                }

                for(int i = 0; i < SIZE; i++)
                {
                    if(f[i] != MEANING)
                    {
                        CHECK_VALUE( log, f[i], MEANING, i );
                    }
                }
            }

            //nd_range, nd_item
            {
                float f[SIZE];
                {
                    cl::sycl::buffer<float,1> buf(&f[0], cl::sycl::range<1>(SIZE));
                    command_group(queue, [&] () {
                        auto a_dev = buf.get_access<access::read_write>();

                        parallel_for<class BASED_ON_ITEM>(
                                    nd_range<1>(range<1>(SIZE)),
                                    [=](nd_item<1> itm)
                        {
                            size_t index = itm.get_global();
                            a_dev[index] = MEANING;
                        });
                    });
                }

                for(int i = 0; i < SIZE; i++)
                {
                    if(f[i] != MEANING)
                    {
                        CHECK_VALUE( log, f[i], MEANING, i );
                    }
                }
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

} /* namespace parallel_for__ */
