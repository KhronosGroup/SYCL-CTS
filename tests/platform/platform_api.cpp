/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_api

namespace platform_api__
{
using namespace sycl_cts;

/** tests the api for cl::sycl::platform
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
            cts_selector selector;
            cl::sycl::platform platform( selector );

            /** check get_devices(device_type = all) method
            */
            auto devices = platform.get_devices();
            if ( typeid( devices ) != typeid(cl::sycl::vector_class<cl::sycl::device>))
            {
                FAIL( log, "get_devices() does not return vector_class<device>" );
            }

            /** check get_devices(device_type) method
            */
            devices = platform.get_devices( cl::sycl::info::device_type::gpu );
            if ( typeid( devices ) != typeid(cl::sycl::vector_class<cl::sycl::device>))
            {
                FAIL( log, "get_devices(device_type) does not return vector_class<device>" );
            }

            /** check has_extensions() method
            */
            auto extensionSupported = platform.has_extension( cl::sycl::string_class( "cl_khr_icd" ) );
            if ( typeid( extensionSupported ) != typeid(bool))
                FAIL( log, "has_extension() does not return bool" );

            /** check get_info() method
            */
            auto platformName = platform.get_info<cl::sycl::info::platform::name>();
            if ( typeid( platformName ) != typeid( cl::sycl::string_class ) )
            {
                FAIL( log, "has_extension() does not string_class" );
            }

            /** check is_host() method
            */
            auto isHost = platform.is_host();
            if ( typeid( isHost ) != typeid(bool))
            {
                FAIL( log, "is_host() does not return bool" );
            }

            /** check get_platforms() static method
            */
            auto platforms = cl::sycl::platform::get_platforms();
            if ( typeid( platforms ) != typeid(cl::sycl::vector_class<cl::sycl::platform>))
            {
                FAIL( log, "get_platforms() does not return vector_class<platform>" );
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

} /* namespace platform_api__ */
