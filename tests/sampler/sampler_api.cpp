/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME sampler_apis

namespace sampler_api__
{
using namespace sycl_cts;

/** tests the api for cl::sycl::sampler
*/
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     */
    virtual void get_info(test_base::info &out) const override
    {
        set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
    }

    /** execute this test
    */
    virtual void run(util::logger &log) override
    {
        try
        {
            cts_selector selector;
            cl::sycl::queue queue( selector );

            queue.submit( [&]( cl::sycl::handler &handler )
            {
                cl::sycl::sampler sampler(false, cl::sycl::sampler_addressing_mode::none, cl::sycl::sampler_filter_mode::nearest);

                /** check is_normalized_corrdinates() method
                */
                auto isNormalizedCoordinates = sampler.is_normalized_coordinates();
                if ( typeid( isNormalizedCoordinates ) != typeid( bool ) )
                {
                    FAIL( log, "is_normalized_coordinates() does not return bool" );
                }

                /** check get_addressing_mode() method
                */
                auto addressingMode = sampler.get_addressing_mode();
                if ( typeid( addressingMode ) != typeid( cl::sycl::sampler_addressing_mode ) )
                {
                    FAIL( log, "get_addressing_mode() does not return sampler_addressing_mode" );
                }

                /** check get_filter_mode() method
                */
                auto filterMode = sampler.get_filter_mode();
                if ( typeid( filterMode ) != typeid( cl::sycl::sampler_filter_mode ) )
                {
                    FAIL( log, "get_filter_mode() does not return sampler_filter_mode" );
                }
            });
        }
        catch (cl::sycl::exception e)
        {
            log_exception(log, e);
            FAIL(log, "a sycl exception was caught");
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace sampler_api__ */
