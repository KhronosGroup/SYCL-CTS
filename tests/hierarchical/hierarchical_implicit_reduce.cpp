/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_reduce

namespace hierarchical_reduce__
{

static const int g_items_1d = 2;
static const int l_items_1d = 2;
static const int g_items_total = g_items_1d * g_items_1d * g_items_1d;
static const int l_items_total = l_items_1d * l_items_1d * l_items_1d;
static const int num_groups = g_items_total / l_items_total;

static const int input_size = 32;

using namespace sycl_cts;
using namespace cl::sycl;

template <typename T>
class sth
{
};

template <typename T>
class sth_else
{
};

template <typename T>
T reduce( T input[input_size], device_selector* selector )
{
    T m_total;
    T m_group_sums[num_groups];

    queue my_queue( *selector );
    buffer<T, 1> input_buf( input, range<1>( input_size ) );
    buffer<T, 1> group_sums_buf( m_group_sums, range<1>( num_groups ) );
    buffer<T, 1> total_buf( &m_total, range<1>( 1 ) );

    my_queue.submit( [&]( handler& cgh )
    {
        accessor<T, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> input_ptr( input_buf, cgh );
        accessor<T, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> group_sums_ptr( group_sums_buf, cgh );
        accessor<T, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> total_ptr( total_buf, cgh );
        cgh.parallel_for_workgroup<class sth<T>>(
                    nd_range<3>( range<3>( g_items_1d, g_items_1d, g_items_1d ),
                                 range<3>( l_items_1d, l_items_1d, l_items_1d )),
                    [=]( group<3> group )
        {
            T local_sums[l_items_total];

            // process items in each work item
            parallel_for_workitem( group, [=, &local_sums]( item<3> item )
                                   {
                int local_id = item.get_global_linear();
                /* Split the array into work-group-size different arrays */
                int values_per_item = ( input_size / num_groups ) / l_items_total;
                int id_start = values_per_item * local_id;
                int id_end = values_per_item * ( local_id + 1 );

                /* Handle the case where the number of input values is not divisible
                * by
                * the number of items. */
                if ( id_end > input_size - 1 )
                    id_end = input_size - 1;

                for ( int i = id_start; i < id_end; i++ )
                    local_sums[i].increment( input_ptr[i] );
            } );

            /* Sum items in each work group */
            for ( int i = 0; i < l_items_total; i++ )
                group_sums_ptr[group.get( 0 )].increment( local_sums[i] );
        });
                     } );

    my_queue.submit( [&]( handler& cgh )
                     {
                         accessor<T, 1, access::read, cl::sycl::access::target::global_buffer> group_sums_ptr( group_sums_buf, cgh );
                         accessor<T, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> total_ptr( total_buf, cgh );

        cgh.single_task<class sth_else<T>>([=]()
        {
            /* Sum items in all work groups */
            for ( int i = 0; i < num_groups; i++ )
            {
                total_ptr[0].value = total_ptr[i].value + group_sums_ptr[i].value;
            }
        });
                     } );

    my_queue.wait_and_throw();

    return m_total;
}

class Adder
{
public:
    Adder()
    {
        value = 0;
    }
    Adder( int val )
    {
        value = val;
    }

    static Adder default_value()
    {
        return Adder( 0 );
    }

    Adder increment( Adder rhs )
    {
        value += rhs.value;
        return *this;
    }

    int value;
};

class Multiplier
{
public:
    Multiplier()
    {
        value = 1;
    }
    Multiplier( int val )
    {
        value = val;
    }

    static Multiplier default_value()
    {
        return Multiplier( 0 );
    }

    Multiplier increment( Multiplier rhs )
    {
        value *= rhs.value;
        return *this;
    }

    int value;
};

/** test cl::sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     */
    virtual void get_info( test_base::info& out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
     */
    virtual void run( util::logger& log ) override
    {
        try
        {
            default_selector sel;
            {
                Adder data[input_size];
                for ( int i = 0; i < input_size; i++ )
                    data[i] = Adder( 2 );

                Adder result = reduce<Adder>( data, &sel );

                int expected_result = input_size * 2;

                if ( result.value != expected_result )
                {
                    FAIL( log, "Incorrect result in Adder" );
                }
            }

            {
                Multiplier data[input_size];
                for ( int i = 0; i < input_size; i++ )
                    data[i] = Multiplier( 2 );

                Multiplier result = reduce<Multiplier>( data, &sel );

                int expected_result = 1;
                for ( int i = 0; i < input_size; ++i )
                    expected_result *= 2;

                if ( result.value != expected_result )
                {
                    FAIL( log, "Incorrect result in Multiplier" );
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

} /* namespace id_api__ */
