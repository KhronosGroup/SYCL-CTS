/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_constructors

namespace device_constructors__
{
using namespace sycl_cts;

/** test cl::sycl::device initialization
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
            cl::sycl::device device;

            cl::sycl::host_selector hs;

            cl::sycl::device host_dev( hs );

            cl::sycl::device device_c( device );
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

} /* namespace device_constructors__ */
