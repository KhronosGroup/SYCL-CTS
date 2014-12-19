/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME range_api

namespace range_api__
{
using namespace sycl_cts;

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
            /** testing gets
             */
            //range 1
            {
                size_t expected = 42;
                cl::sycl::range<1> range( expected );
                CHECK_VALUE( log, range.get( 0 ), expected, 0 );
                CHECK_VALUE( log, range[0], expected, 0 );
            }

            //range 2
            {
                size_t expected[] = { 42, 99 };
                cl::sycl::range<2> range( expected[0], expected[1] );

                for(int i = 0; i < 2; i++)
                {
                    CHECK_VALUE( log, range.get( i ), expected[i], i );
                    CHECK_VALUE( log, range[i], expected[i], i );
                }
            }

            //range 3
            {
                size_t expected[] = { 42, 99, 129 };
                cl::sycl::range<3> range( expected[0], expected[1], expected[2] );

                for(int i = 0; i < 3; i++)
                {
                    CHECK_VALUE( log, range.get( i ), expected[i], i );
                    CHECK_VALUE( log, range[i], expected[i], i );
                }
            }

            /** testing operators
             */
            //range 1
            {
                size_t expectedOne = 42;
                size_t expectedTwo = 2;
                cl::sycl::range<1> rangeOne( expectedOne );
                cl::sycl::range<1> rangeTwo( expectedTwo );
                //operator +
                {
                    cl::sycl::range<1> range = rangeOne + rangeTwo;
                    CHECK_VALUE( log, range.get( 0 ), expectedOne + expectedTwo, 0 );
                    CHECK_VALUE( log, range[0], expectedOne + expectedTwo, 0 );
                }
                //operator -
                {
                    cl::sycl::range<1> range = rangeOne - rangeTwo;
                    CHECK_VALUE( log, range.get( 0 ), expectedOne - expectedTwo, 0 );
                    CHECK_VALUE( log, range[0], expectedOne - expectedTwo, 0 );
                }
                //operator /
                {
                    cl::sycl::range<1> range = rangeOne / rangeTwo;
                    CHECK_VALUE( log, range.get( 0 ), expectedOne / expectedTwo, 0 );
                    CHECK_VALUE( log, range[0], expectedOne / expectedTwo, 0 );
                }
                //operator *
                {
                    cl::sycl::range<1> range = rangeTwo * rangeOne;
                    CHECK_VALUE( log, range.get( 0 ), expectedOne * expectedTwo, 0 );
                    CHECK_VALUE( log, range[0], expectedOne * expectedTwo , 0 );
                }
            } //range 1

            //range 2
            {
                const int elements = 2;
                size_t expectedOne[] = {42, 128 };
                size_t expectedTwo[] = { 2, 16 };
                cl::sycl::range<elements> rangeOne( expectedOne[0], expectedOne[1] );
                cl::sycl::range<elements> rangeTwo( expectedTwo[0], expectedTwo[1] );
                //operator +
                {
                    cl::sycl::range<elements> range = rangeOne + rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE( log, range.get( i ), expectedOne[i] + expectedTwo[i], i );
                        CHECK_VALUE( log, range[i], expectedOne[i] + expectedTwo[i], i );
                    }
                }
                //operator -
                {
                    cl::sycl::range<elements> range = rangeOne - rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE( log, range.get( i ), expectedOne[i] - expectedTwo[i], i );
                        CHECK_VALUE( log, range[i], expectedOne[i] - expectedTwo[i], i );
                    }
                }
                //operator /
                {
                    cl::sycl::range<elements> range = rangeOne / rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE( log, range.get( i ), expectedOne[i] / expectedTwo[i], i );
                        CHECK_VALUE( log, range[i], expectedOne[i] / expectedTwo[i], i );
                    }
                }
                //operator *
                {
                    cl::sycl::range<elements> range = rangeOne * rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE( log, range.get( i ), expectedOne[i] * expectedTwo[i], i );
                        CHECK_VALUE( log, range[i], expectedOne[i] * expectedTwo[i], i );
                    }
                }
            } //range 2

            //range 3
            {
                const int elements = 3;
                size_t expectedOne[] = {42, 128, 256 };
                size_t expectedTwo[] = { 2, 16, 128 };
                cl::sycl::range<elements> rangeOne( expectedOne[0], expectedOne[1], expectedOne[2] );
                cl::sycl::range<elements> rangeTwo( expectedTwo[0], expectedTwo[1], expectedTwo[2] );
                //operator +
                {
                    cl::sycl::range<elements> range = rangeOne + rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE(log, range.get( i ), expectedOne[i] + expectedTwo[i], i );
                        CHECK_VALUE(log, range[i], expectedOne[i] + expectedTwo[i], i );
                    }
                }
                //operator -
                {
                    cl::sycl::range<elements> range = rangeOne - rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE(log, range.get( i ), expectedOne[i] - expectedTwo[i], i );
                        CHECK_VALUE(log, range[i], expectedOne[i] - expectedTwo[i], i );
                    }
                }
                //operator /
                {
                    cl::sycl::range<elements> range = rangeOne / rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE(log, range.get( i ), expectedOne[i] / expectedTwo[i], i );
                        CHECK_VALUE(log, range[i], expectedOne[i] / expectedTwo[i], i );
                    }
                }
                //operator *
                {
                    cl::sycl::range<elements> range = rangeOne * rangeTwo;
                    for(int i = 0; i < elements; i++)
                    {
                        CHECK_VALUE(log, range.get( i ), expectedOne[i] * expectedTwo[i], i );
                        CHECK_VALUE(log, range[i], expectedOne[i] * expectedTwo[i], i );
                    }
                }
            } //range 3

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

} /* namespace range_api__ */
