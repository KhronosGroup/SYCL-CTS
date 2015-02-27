/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_constructors

namespace device_constructors__
{
using namespace sycl_cts;

/** tests the constructors for cl::sycl::device
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
        try
        {
            /** check default constructor and destructor
            */
            {
                cl::sycl::device device;

                if ( !device.is_host() )
                {
                    FAIL( log, "device was not constructed correctly" );
                }

                if ( device.get() != 0 )
                {
                    FAIL( log, "device was not constructed correctly" );
                }
            }

            /** check (device_selector) constructor
            */
            {
                cts_selector selector;
                cl::sycl::device device( selector );

                if ( device.is_host() )
                {
                    FAIL( log, "device was not constructed correctly" );
                }

                if ( device.get() == 0 )
                {
                    FAIL( log, "device was not constructed correctly" );
                }
            }

            /** check copy constructor
            */
            {
                cts_selector selector;
                cl::sycl::device deviceA( selector );
                cl::sycl::device deviceB( deviceA );

                if ( deviceA.get() != deviceB.get() )
                {
                    FAIL( log, "device was not copied correctly" );
                }
            }

            /** check assignment operator
            */
            {
                cts_selector selector;
                cl::sycl::device deviceA( selector );
                cl::sycl::device deviceB = deviceA;

                if ( deviceA.get() != deviceB.get() )
                {
                    FAIL( log, "device was not assigned correctly" );
                }
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "a sycl exception was caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_constructors__ */
