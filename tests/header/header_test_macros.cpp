/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME header_test_2
#define TEST_CATEGORY header

namespace header_test_2__
{
using namespace sycl_cts;

/** test SYCL header for macro definitions
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

    /** execute this test
     *  @return, one of test_result enum
     */
    virtual void run( util::logger &log )
    {

// test for existence of CL_SYCL_LANGUAGE_VERSION
#if !defined( CL_SYCL_LANGUAGE_VERSION )
#define TEST_FAIL
        log.note( "CL_SYCL_LANGUAGE_VERSION not present" )
#else
        log.note( "CL_SYCL_LANGUAGE_VERSION = %d", (int)CL_SYCL_LANGUAGE_VERSION );
#endif

// test for the existence of default vector class
#if !defined( VECTOR_CLASS )
#define TEST_FAIL
        log.note( "VECTOR_CLASS not present" )
#endif

// test for the existence of default string class
#if !defined( STRING_CLASS )
#define TEST_FAIL
        log.note( "STRING_CLASS not present" )
#endif

#if defined( TEST_FAIL )
        FAIL( log, "" );
#endif
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace header_test_2__ */
