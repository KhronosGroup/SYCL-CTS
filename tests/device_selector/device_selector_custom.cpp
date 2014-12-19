/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_selector_custom

namespace device_selector_custom__
{
using namespace sycl_cts;

/** device selection functor
 */
class custom_selector : public cl::sycl::device_selector
{
public:
    mutable bool m_hasBeenCalled;

    custom_selector()
        : m_hasBeenCalled( false )
    {
    }

    /** device selection operator
     *  return <  0  : device will never be selected
     *  return >= 0  : positive device rating
     */
    virtual int operator()( const cl::sycl::device &dev ) const
    {
        m_hasBeenCalled = true;
        return 1;
    }
};

/** check that we can use a custom device selector
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
            // instantiate a custom device selector
            custom_selector selector;

            cl::sycl::platform platform( selector );

            /* check our device selector was used */
            if ( !selector.m_hasBeenCalled )
                FAIL( log, "custom selector never used!" );
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_selector_custom__ */
