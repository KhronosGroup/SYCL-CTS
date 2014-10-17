/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <CL/sycl.hpp>

#include "../common/common.h"

#define TEST_NAME buffer_api

namespace sycl_cts
{

    /** test cl::sycl::buffer api
    */
    class TEST_NAME
        : public util::test_base
    {
    public:

        /** return information about this test
        *  @param info, test_base::info structure as output
        */
        virtual void get_info(test_base::info & out) const
        {
            set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
        }

        /** execute the test
        *  @param log, test transcript logging class
        */
        virtual void run(util::logger & log)
        {
            try
            {
                const size_t range_size = 32;
                int data[range_size] = { 0 };
                cl::sycl::range<1> range(range_size);
#if ENABLE_FULL_TEST
                cl::sycl::buffer<int, 1> buf(data, range);
                auto ret_range = buf.get_range();
                if (typeid(ret_range) != typeid(cl::sycl::range<1>))
                {
                    FAIL(log, "cl::sycl::buffer::get_range does not return " \
                        "cl::sycl::range!");
                }

                size_t count = buf.get_count();
                size_t size  = buf.get_size ();
                auto access  = buf.get_access<cl::sycl::access::read_write>();
#endif
            }
            catch (cl::sycl::sycl_error e)
            {
                log_exception(log, e);
                FAIL(log, "");
            }
        }

    };

    // construction of this proxy will register the above test
    static util::test_proxy<TEST_NAME> proxy;

}; // sycl_cts
