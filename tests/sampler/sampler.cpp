/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME sampler_constructors

namespace sampler__
{
using namespace sycl_cts;

class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
    *  @param info, test_base::info structure as output
    */
    virtual void get_info(test_base::info &out) const
    {
        set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
    }

    /** execute the test
    *  @param log, test transcript logging class
    */
    virtual void run(util::logger &log)
    {
        try
        {
            bool normalise = true;
            cl::sycl::sampler s( normalise,
                sampler_addressing_mode::SYCL_SAMPLER_ADDRESS_CLAMP,
                sampler_filter_mode::SYCL_SAMPLER_FILTER_LINEAR );

            auto address_mode = s.get_address();
            if ( typeid( address_mode ) != typeid( cl_addressing_mode ) )
                FAIL( log, "sampler::get_address() does not return "
                           "cl_addressing_mode" );
            if ( address_mode != CL_ADDRESS_CLAMP )
                FAIL( log, "sampler::get_address() returned wrong "
                           "address mode value" );

            auto filter_mode = s.get_filter();
            if ( typeid( filter_mode ) != typeid( cl_filter_mode ) )
                FAIL( log, "sampler::get_filter() does not return "
                           "cl_filter_mode" );
            if ( address_mode != CL_FILTER_LINEAR )
                FAIL( log, "sampler::get_filter() returned wrong "
                           "filter mode value" );

            auto cl_s = s.get_opencl_sampler_object();
            if ( typeid( cl_s ) != typeid( cl_sampler ) )
                FAIL(log, "sampler::get_opencl_sampler_object() "
                          "does not return cl_sampler" );

            cl_int error = clReleaseSampler( cl_s );
            if( error != CL_SUCCESS )
                FAIL( log, "cl_sampler object was not valid" );
        }
        catch (cl::sycl::exception e)
        {
            log_exception(log, e);
            FAIL(log, "sycl exception caught");
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace sampler__ */
