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

#define CHECK_EXISTS(EXCEPTION_NAME)\
        if (!std::is_class<EXCEPTION_NAME>::value) {\
          FAIL(log, "EXCEPTION_NAME is not defined as a class");\
        }

        CHECK_EXISTS(exception);
        CHECK_EXISTS(cl_exception);
        CHECK_EXISTS(async_exception);
        CHECK_EXISTS(runtime_error);
        CHECK_EXISTS(kernel_error);
        CHECK_EXISTS(nd_range_error);
        CHECK_EXISTS(event_error);
        CHECK_EXISTS(invalid_parameter_error);
        CHECK_EXISTS(device_error);
        CHECK_EXISTS(compile_program_error);
        CHECK_EXISTS(link_program_error);
        CHECK_EXISTS(invalid_object_error);
        CHECK_EXISTS(memory_allocation_error);
        CHECK_EXISTS(platform_error);
        CHECK_EXISTS(profiling_error);
#undef CHECK_EXISTS

        /* Check that exception_list exists */
        if (!std::is_class<exception_list>::value) {
            FAIL(log, "Exception list is not defined as a class");
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace header_test_2__ */
