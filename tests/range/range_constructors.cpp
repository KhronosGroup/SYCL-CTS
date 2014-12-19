/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME range_constructors

namespace range_constructors__
{
using namespace sycl_cts;

/** test cl::sycl::range initialization
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
            cl::sycl::range<1> rangeOne( 1 );
            cl::sycl::range<2> rangeTwo( 1, 2 );
            cl::sycl::range<3> rangeThree( 1, 2, 3 );
            cl::sycl::range<1> rangeFour( rangeOne );
            cl::sycl::range<2> rangeFive( rangeTwo );
            cl::sycl::range<3> rangeSix( rangeThree );
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

} /* namespace range_constructors__ */
