/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#define CL_SYCL_NO_STD_UNIQUE_PTR

#include "../common/common.h"

#define TEST_NAME std_class_unique_ptr

namespace std_class_unique_ptr__
{
using namespace sycl_cts;

// if the CL_SYCL_NO_STD_UNIQUE_PTR define works there should be no clash with the definition below
class unique_ptr_class {
public:

    bool my_method( ) {
        return true;
    }

};

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

        unique_ptr_class my_class;
        if ( !my_class.my_method( ) )
            FAIL( log, "" );
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace std_class_shared_ptr__ */
