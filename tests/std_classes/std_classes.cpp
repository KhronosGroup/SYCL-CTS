/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME std_classes

namespace std_classes__
{
using namespace sycl_cts;

bool test_vector_class( util::logger & log )
{
    cl::sycl::vector_class<int> sycl_vec;

    const int num_its = 10;

    for ( int i = 0; i < num_its; i++ ) {
        sycl_vec.push_back( 0 );
    }

    if ( sycl_vec.size( ) != num_its( ) )
        FAIL( log, "vector size incorrect" );

    for ( int i = 0; i < num_its; i++ ) {
        if ( sycl_vec[i] != i ) {
            FAIL( log, "element index incorrect" );
        }
    }
    return log.has_failed( );
}

bool test_string_class( util::logger & log )
{
    cl::sycl::string_class sycl_string;

    return log.has_failed( );
}

bool test_function_class( util::logger & log )
{
    cl::sycl::function_class sycl_function;

    return log.has_failed( );
}

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

        if ( !test_vector_class( log ) )
            return;
        if ( !test_string_class( log ) )
            return;
        if ( !test_function_class( log ) )
            return;
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace std_classes__ */
