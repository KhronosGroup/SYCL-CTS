/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:  (c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "./../../util/math_helper.h"

#define TEST_NAME vector_logical_operators

namespace vector_logical_operators__
{
using namespace cl::sycl;
using namespace sycl_cts;

/** test SYCL header for compilation
 */
template <typename T>
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info(test_base::info &out) const
    {
        const char *l_name = "";
#define MAKENAME( X )                          \
    if ( typeid( T ) == typeid( X ) )          \
    {                                          \
        l_name = TOSTRING( TEST_NAME ) "_" #X; \
    }
        MAKENAME(float);
        MAKENAME(float2);
        MAKENAME(float3);
        MAKENAME(float4);
        MAKENAME(float8);
        MAKENAME(float16);
#undef MAKENAME
        set_test_info(out, l_name, TEST_FILE);
    }

    /** execute this test
     *  @return, one of test_result enum
     */
    virtual void run(util::logger &log)
    {
        try
        {
            T lhs_data;
            math::fill(lhs_data, 1);
            T rhs_data(0);
            math::fill(rhs_data, 1);

            /* verify */
            if (lhs_data == rhs_data)
            {
                log.note("logical operator == functioning properly");
            }
            else
            {
                FAIL(log, "logical operator == NOT functioning properly");
            }

            math::fill(rhs_data, 2);

            /* verify */
            if (lhs_data != rhs_data)
            {
                log.note("logical operator != functioning properly");
            }
            else
            {
                FAIL(log, "logical operator != NOT functioning properly");
            }

        }
        catch (cl::sycl::exception e)
        {
            log_exception(log, e);
            FAIL(log, "sycl exception caught");
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME<float2>>  proxy2;
util::test_proxy<TEST_NAME<float3>>  proxy3;
util::test_proxy<TEST_NAME<float4>>  proxy4;
util::test_proxy<TEST_NAME<float8>>  proxy8;
util::test_proxy<TEST_NAME<float16>> proxy16;

}; /* namespace vector_logical_operators__ */
