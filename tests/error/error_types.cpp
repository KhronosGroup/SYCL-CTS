/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME error_types

namespace error_types
{
using namespace sycl_cts;

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

    /** execute this test
     */
    virtual void run( util::logger &log ) override
    {
        using namespace cl::sycl;

        exception         sycl_exception;
        cl_exception      sycl_cl_exception;
        async_exception   async_exception;
        exception_list    sycl_exception_list;
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace header_test_2__ */
