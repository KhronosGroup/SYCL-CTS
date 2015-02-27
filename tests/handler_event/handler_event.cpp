/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_handler

namespace TEST_NAMESPACE
{
using namespace sycl_cts;
using namespace cl::sycl;

/** test SYCL header for compilation
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

    /** execute this test
    */
    virtual void run( util::logger &log ) override
    {
        cts_selector sel;
        queue q( sel );

        handler_event hev = q.submit( [&]( handler & cgh )
        {
            cgh.single_task( [=]()
            {
                int  a = 0;
            } );
        } );

        event kernel_event = hev.get_kernel();
        event complete_event = hev.get_complete();
        event end_event = hev.get_end();
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace header_test_1 */
