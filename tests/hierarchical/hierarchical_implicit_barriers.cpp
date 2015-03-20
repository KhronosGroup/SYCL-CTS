/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_implicit_barriers

namespace hierarchical_implicit_barriers__
{

static const unsigned int g_items_1d = 8;
static const unsigned int g_items_2d = 4;
static const unsigned int g_items_3d = 2;
static const unsigned int l_items_1d = 4;
static const unsigned int l_items_2d = 2;
static const unsigned int l_items_3d = 1;
static const unsigned int gr_range_1d = ( g_items_1d / l_items_1d );
static const unsigned int gr_range_2d = ( g_items_2d / l_items_2d );
static const unsigned int gr_range_3d = ( g_items_3d / l_items_3d );
static const int g_items_total = ( g_items_1d * g_items_2d * g_items_3d );
static const unsigned int l_items_total = ( l_items_1d * l_items_2d * l_items_3d );
static const unsigned int gr_range_total = ( g_items_total / l_items_total );

using namespace sycl_cts;
using namespace cl::sycl;

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
            int input_data[g_items_total];

            cl::sycl::default_selector sel;
            cl::sycl::queue testQueue ( sel );


            for( size_t i = 0; i < g_items_total; i++ )
            {
                input_data[i] = i;
            }
            {
                buffer<int, 1> input_buffer( input_data, range<1> ( g_items_total ) );


                testQueue.submit([&]( handler & cgh )
                {
                    auto my_range = nd_range<3> (
                                range<3>( g_items_1d, g_items_2d, g_items_3d ),
                                range<3>( l_items_1d, l_items_2d, l_items_3d )
                                );

                    auto input_ptr = input_buffer.get_access<cl::sycl::access::mode::read>( cgh );

                    accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_ptr (
                                range<1>( l_items_total ), cgh );
                    cgh.parallel_for_work_group<class hierarchical_implicit_barriers>(
                                my_range, [=]( group<3> group )
                    {
                        parallel_for_work_item( group, [&]( item<3> item )
                        {
                            int global_id = item.get_global_linear( );
                            int global_size = item.get_global_linear_range();
                            int local_id = item.get_local_linear();

                           int inverted_val = ( global_size -1 ) - input_ptr[global_id];

                           local_ptr[local_id] = inverted_val;
                        });

                        parallel_for_work_item( group, [&]( item<3> item )
                        {
                            int global_id = item.get_global_linear( );
                            int local_id = item.get_local_linear( );

                           input_ptr[global_id] = local_ptr[local_id];
                        });

                    });
                });
            }

            for ( int i = 0; i < g_items_total; i++ )
            {
                if ( input_data[( g_items_total - 1 ) - i] != i )
                {
                    FAIL(log, "Values not equal.");
                }
            }

            testQueue.wait_and_throw();

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

