
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_of_vector.py
//
************************************************************************************/
/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:  (c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#define SYCL_SIMPLE_SWIZZLES
#include "../common/common.h"
#include "./../../util/math_helper.h"

using namespace cl::sycl;

#define TEST_NAME vector_of_vector_char8
#define KERNEL_NAME cKernel_vector_of_vector
#define NUM_TESTS 109

/* CONSTRUCTOR TESTS SPECIAL MACROS FOR EASIER GENERATION AND VERIFICATION*/
#define V1S0 1
#define V2S0 2
#define V2S1 3
#define V3S0 4
#define V3S1 5
#define V3S2 6
#define V4S0 7
#define V4S1 8
#define V4S2 9
#define V4S3 10
#define V8S0 11
#define V8S1 12
#define V8S2 13
#define V8S3 14
#define V8S4 15
#define V8S5 16
#define V8S6 17
#define V8S7 18
#define V16S0 19
#define V16S1 20
#define V16S2 21
#define V16S3 22
#define V16S4 23
#define V16S5 24
#define V16S6 25
#define V16S7 26
#define V16S8 27
#define V16S9 28
#define V16S10 29
#define V16S11 30
#define V16S12 31
#define V16S13 32
#define V16S14 33
#define V16S15 34

#define V1 V1S0
#define V2 V2S0, V2S1
#define V3 V3S0, V3S1, V3S2
#define V4 V4S0, V4S1, V4S2, V4S3
#define V8 V8S0, V8S1, V8S2, V8S3, V8S4, V8S5, V8S6, V8S7
#define V16 V16S0, V16S1, V16S2, V16S3, V16S4, V16S5, V16S6, V16S7, V16S8, V16S9, V16S10, V16S11, V16S12, V16S13, V16S14, V16S15

#define char1 char
#define CONSTRUCTOR_TEST(INDEX, ...) m_o[INDEX] = T(__VA_ARGS__);
#define VERIFY_EQUALS(INDEX, ...) if( all( odata[INDEX] != T(__VA_ARGS__) ) ){ FAIL(log, "results don't match"); }


namespace TEST_NAME
{
using namespace sycl_cts;

/** kernel functor
 */
template <typename T>
class KERNEL_NAME
{
protected:
    typedef accessor<T, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> t_writeAccess;

    t_writeAccess m_o; /* output     */

public:
    KERNEL_NAME(t_writeAccess out_)
        : m_o(out_)
    {
    }

    void operator()(item<1> item)
    {
        char1  v1  (V1S0);
    	char2  v2  (V2S0,  V2S1);
		char3  v3  (V3S0,  V3S1,  V3S2);
		char4  v4  (V4S0,  V4S1,  V4S2,  V4S3);
		char8  v8  (V8S0,  V8S1,  V8S2,  V8S3,  V8S4,  V8S5,  V8S6,  V8S7);
		char16 v16 (V16S0, V16S1, V16S2, V16S3, V16S4, V16S5, V16S6, V16S7, V16S8, V16S9, V16S10, V16S11, V16S12, V16S13, V16S14, V16S15);

        /* MACROS GENERATED FROM PYTHON SCRIPT*/
        
        CONSTRUCTOR_TEST( 0, v8 )
        CONSTRUCTOR_TEST( 1, v4, v4 )
        CONSTRUCTOR_TEST( 2, v1, v3, v4 )
        CONSTRUCTOR_TEST( 3, v1, v4, v3 )
        CONSTRUCTOR_TEST( 4, v2, v2, v4 )
        CONSTRUCTOR_TEST( 5, v2, v3, v3 )
        CONSTRUCTOR_TEST( 6, v2, v4, v2 )
        CONSTRUCTOR_TEST( 7, v3, v1, v4 )
        CONSTRUCTOR_TEST( 8, v3, v2, v3 )
        CONSTRUCTOR_TEST( 9, v3, v3, v2 )
        CONSTRUCTOR_TEST( 10, v3, v4, v1 )
        CONSTRUCTOR_TEST( 11, v4, v1, v3 )
        CONSTRUCTOR_TEST( 12, v4, v2, v2 )
        CONSTRUCTOR_TEST( 13, v4, v3, v1 )
        CONSTRUCTOR_TEST( 14, v1, v1, v2, v4 )
        CONSTRUCTOR_TEST( 15, v1, v1, v3, v3 )
        CONSTRUCTOR_TEST( 16, v1, v1, v4, v2 )
        CONSTRUCTOR_TEST( 17, v1, v2, v1, v4 )
        CONSTRUCTOR_TEST( 18, v1, v2, v2, v3 )
        CONSTRUCTOR_TEST( 19, v1, v2, v3, v2 )
        CONSTRUCTOR_TEST( 20, v1, v2, v4, v1 )
        CONSTRUCTOR_TEST( 21, v1, v3, v1, v3 )
        CONSTRUCTOR_TEST( 22, v1, v3, v2, v2 )
        CONSTRUCTOR_TEST( 23, v1, v3, v3, v1 )
        CONSTRUCTOR_TEST( 24, v1, v4, v1, v2 )
        CONSTRUCTOR_TEST( 25, v1, v4, v2, v1 )
        CONSTRUCTOR_TEST( 26, v2, v1, v1, v4 )
        CONSTRUCTOR_TEST( 27, v2, v1, v2, v3 )
        CONSTRUCTOR_TEST( 28, v2, v1, v3, v2 )
        CONSTRUCTOR_TEST( 29, v2, v1, v4, v1 )
        CONSTRUCTOR_TEST( 30, v2, v2, v1, v3 )
        CONSTRUCTOR_TEST( 31, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 32, v2, v2, v3, v1 )
        CONSTRUCTOR_TEST( 33, v2, v3, v1, v2 )
        CONSTRUCTOR_TEST( 34, v2, v3, v2, v1 )
        CONSTRUCTOR_TEST( 35, v2, v4, v1, v1 )
        CONSTRUCTOR_TEST( 36, v3, v1, v1, v3 )
        CONSTRUCTOR_TEST( 37, v3, v1, v2, v2 )
        CONSTRUCTOR_TEST( 38, v3, v1, v3, v1 )
        CONSTRUCTOR_TEST( 39, v3, v2, v1, v2 )
        CONSTRUCTOR_TEST( 40, v3, v2, v2, v1 )
        CONSTRUCTOR_TEST( 41, v3, v3, v1, v1 )
        CONSTRUCTOR_TEST( 42, v4, v1, v1, v2 )
        CONSTRUCTOR_TEST( 43, v4, v1, v2, v1 )
        CONSTRUCTOR_TEST( 44, v4, v2, v1, v1 )
        CONSTRUCTOR_TEST( 45, v1, v1, v1, v1, v4 )
        CONSTRUCTOR_TEST( 46, v1, v1, v1, v2, v3 )
        CONSTRUCTOR_TEST( 47, v1, v1, v1, v3, v2 )
        CONSTRUCTOR_TEST( 48, v1, v1, v1, v4, v1 )
        CONSTRUCTOR_TEST( 49, v1, v1, v2, v1, v3 )
        CONSTRUCTOR_TEST( 50, v1, v1, v2, v2, v2 )
        CONSTRUCTOR_TEST( 51, v1, v1, v2, v3, v1 )
        CONSTRUCTOR_TEST( 52, v1, v1, v3, v1, v2 )
        CONSTRUCTOR_TEST( 53, v1, v1, v3, v2, v1 )
        CONSTRUCTOR_TEST( 54, v1, v1, v4, v1, v1 )
        CONSTRUCTOR_TEST( 55, v1, v2, v1, v1, v3 )
        CONSTRUCTOR_TEST( 56, v1, v2, v1, v2, v2 )
        CONSTRUCTOR_TEST( 57, v1, v2, v1, v3, v1 )
        CONSTRUCTOR_TEST( 58, v1, v2, v2, v1, v2 )
        CONSTRUCTOR_TEST( 59, v1, v2, v2, v2, v1 )
        CONSTRUCTOR_TEST( 60, v1, v2, v3, v1, v1 )
        CONSTRUCTOR_TEST( 61, v1, v3, v1, v1, v2 )
        CONSTRUCTOR_TEST( 62, v1, v3, v1, v2, v1 )
        CONSTRUCTOR_TEST( 63, v1, v3, v2, v1, v1 )
        CONSTRUCTOR_TEST( 64, v1, v4, v1, v1, v1 )
        CONSTRUCTOR_TEST( 65, v2, v1, v1, v1, v3 )
        CONSTRUCTOR_TEST( 66, v2, v1, v1, v2, v2 )
        CONSTRUCTOR_TEST( 67, v2, v1, v1, v3, v1 )
        CONSTRUCTOR_TEST( 68, v2, v1, v2, v1, v2 )
        CONSTRUCTOR_TEST( 69, v2, v1, v2, v2, v1 )
        CONSTRUCTOR_TEST( 70, v2, v1, v3, v1, v1 )
        CONSTRUCTOR_TEST( 71, v2, v2, v1, v1, v2 )
        CONSTRUCTOR_TEST( 72, v2, v2, v1, v2, v1 )
        CONSTRUCTOR_TEST( 73, v2, v2, v2, v1, v1 )
        CONSTRUCTOR_TEST( 74, v2, v3, v1, v1, v1 )
        CONSTRUCTOR_TEST( 75, v3, v1, v1, v1, v2 )
        CONSTRUCTOR_TEST( 76, v3, v1, v1, v2, v1 )
        CONSTRUCTOR_TEST( 77, v3, v1, v2, v1, v1 )
        CONSTRUCTOR_TEST( 78, v3, v2, v1, v1, v1 )
        CONSTRUCTOR_TEST( 79, v4, v1, v1, v1, v1 )
        CONSTRUCTOR_TEST( 80, v1, v1, v1, v1, v1, v3 )
        CONSTRUCTOR_TEST( 81, v1, v1, v1, v1, v2, v2 )
        CONSTRUCTOR_TEST( 82, v1, v1, v1, v1, v3, v1 )
        CONSTRUCTOR_TEST( 83, v1, v1, v1, v2, v1, v2 )
        CONSTRUCTOR_TEST( 84, v1, v1, v1, v2, v2, v1 )
        CONSTRUCTOR_TEST( 85, v1, v1, v1, v3, v1, v1 )
        CONSTRUCTOR_TEST( 86, v1, v1, v2, v1, v1, v2 )
        CONSTRUCTOR_TEST( 87, v1, v1, v2, v1, v2, v1 )
        CONSTRUCTOR_TEST( 88, v1, v1, v2, v2, v1, v1 )
        CONSTRUCTOR_TEST( 89, v1, v1, v3, v1, v1, v1 )
        CONSTRUCTOR_TEST( 90, v1, v2, v1, v1, v1, v2 )
        CONSTRUCTOR_TEST( 91, v1, v2, v1, v1, v2, v1 )
        CONSTRUCTOR_TEST( 92, v1, v2, v1, v2, v1, v1 )
        CONSTRUCTOR_TEST( 93, v1, v2, v2, v1, v1, v1 )
        CONSTRUCTOR_TEST( 94, v1, v3, v1, v1, v1, v1 )
        CONSTRUCTOR_TEST( 95, v2, v1, v1, v1, v1, v2 )
        CONSTRUCTOR_TEST( 96, v2, v1, v1, v1, v2, v1 )
        CONSTRUCTOR_TEST( 97, v2, v1, v1, v2, v1, v1 )
        CONSTRUCTOR_TEST( 98, v2, v1, v2, v1, v1, v1 )
        CONSTRUCTOR_TEST( 99, v2, v2, v1, v1, v1, v1 )
        CONSTRUCTOR_TEST( 100, v3, v1, v1, v1, v1, v1 )
        CONSTRUCTOR_TEST( 101, v1, v1, v1, v1, v1, v1, v2 )
        CONSTRUCTOR_TEST( 102, v1, v1, v1, v1, v1, v2, v1 )
        CONSTRUCTOR_TEST( 103, v1, v1, v1, v1, v2, v1, v1 )
        CONSTRUCTOR_TEST( 104, v1, v1, v1, v2, v1, v1, v1 )
        CONSTRUCTOR_TEST( 105, v1, v1, v2, v1, v1, v1, v1 )
        CONSTRUCTOR_TEST( 106, v1, v2, v1, v1, v1, v1, v1 )
        CONSTRUCTOR_TEST( 107, v2, v1, v1, v1, v1, v1, v1 )
        CONSTRUCTOR_TEST( 108, v1, v1, v1, v1, v1, v1, v1, v1 )
        
        /* MACROS GENERATED FROM PYTHON SCRIPT*/
    }
};

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
        MAKENAME(char2);
        MAKENAME(char3);
        MAKENAME(char4);
        MAKENAME(char8);
        MAKENAME(char16);
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
            T odata[NUM_TESTS];

			// construct the cts default selector
            cts_selector selector;

            /* create command queue */
            queue l_queue(selector);

            buffer<T, 1> obuf(odata, range<1> ( NUM_TESTS ) );

            /* add command to queue */
            l_queue.submit( [&]( handler& cgh )
            {
                auto optr = obuf.template get_access<cl::sycl::access::mode::write>( cgh );

                /* instantiate the kernel */
                auto kern = KERNEL_NAME<T>( optr );

                /* execute the kernel */
                cgh.parallel_for( nd_range<1>( range<1>( 1 ), range<1>( 1 )), kern );
            } );

            /* MACROS GENERATED FROM PYTHON SCRIPT*/
            
            VERIFY_EQUALS( 0, V8 )
            VERIFY_EQUALS( 1, V4, V4 )
            VERIFY_EQUALS( 2, V1, V3, V4 )
            VERIFY_EQUALS( 3, V1, V4, V3 )
            VERIFY_EQUALS( 4, V2, V2, V4 )
            VERIFY_EQUALS( 5, V2, V3, V3 )
            VERIFY_EQUALS( 6, V2, V4, V2 )
            VERIFY_EQUALS( 7, V3, V1, V4 )
            VERIFY_EQUALS( 8, V3, V2, V3 )
            VERIFY_EQUALS( 9, V3, V3, V2 )
            VERIFY_EQUALS( 10, V3, V4, V1 )
            VERIFY_EQUALS( 11, V4, V1, V3 )
            VERIFY_EQUALS( 12, V4, V2, V2 )
            VERIFY_EQUALS( 13, V4, V3, V1 )
            VERIFY_EQUALS( 14, V1, V1, V2, V4 )
            VERIFY_EQUALS( 15, V1, V1, V3, V3 )
            VERIFY_EQUALS( 16, V1, V1, V4, V2 )
            VERIFY_EQUALS( 17, V1, V2, V1, V4 )
            VERIFY_EQUALS( 18, V1, V2, V2, V3 )
            VERIFY_EQUALS( 19, V1, V2, V3, V2 )
            VERIFY_EQUALS( 20, V1, V2, V4, V1 )
            VERIFY_EQUALS( 21, V1, V3, V1, V3 )
            VERIFY_EQUALS( 22, V1, V3, V2, V2 )
            VERIFY_EQUALS( 23, V1, V3, V3, V1 )
            VERIFY_EQUALS( 24, V1, V4, V1, V2 )
            VERIFY_EQUALS( 25, V1, V4, V2, V1 )
            VERIFY_EQUALS( 26, V2, V1, V1, V4 )
            VERIFY_EQUALS( 27, V2, V1, V2, V3 )
            VERIFY_EQUALS( 28, V2, V1, V3, V2 )
            VERIFY_EQUALS( 29, V2, V1, V4, V1 )
            VERIFY_EQUALS( 30, V2, V2, V1, V3 )
            VERIFY_EQUALS( 31, V2, V2, V2, V2 )
            VERIFY_EQUALS( 32, V2, V2, V3, V1 )
            VERIFY_EQUALS( 33, V2, V3, V1, V2 )
            VERIFY_EQUALS( 34, V2, V3, V2, V1 )
            VERIFY_EQUALS( 35, V2, V4, V1, V1 )
            VERIFY_EQUALS( 36, V3, V1, V1, V3 )
            VERIFY_EQUALS( 37, V3, V1, V2, V2 )
            VERIFY_EQUALS( 38, V3, V1, V3, V1 )
            VERIFY_EQUALS( 39, V3, V2, V1, V2 )
            VERIFY_EQUALS( 40, V3, V2, V2, V1 )
            VERIFY_EQUALS( 41, V3, V3, V1, V1 )
            VERIFY_EQUALS( 42, V4, V1, V1, V2 )
            VERIFY_EQUALS( 43, V4, V1, V2, V1 )
            VERIFY_EQUALS( 44, V4, V2, V1, V1 )
            VERIFY_EQUALS( 45, V1, V1, V1, V1, V4 )
            VERIFY_EQUALS( 46, V1, V1, V1, V2, V3 )
            VERIFY_EQUALS( 47, V1, V1, V1, V3, V2 )
            VERIFY_EQUALS( 48, V1, V1, V1, V4, V1 )
            VERIFY_EQUALS( 49, V1, V1, V2, V1, V3 )
            VERIFY_EQUALS( 50, V1, V1, V2, V2, V2 )
            VERIFY_EQUALS( 51, V1, V1, V2, V3, V1 )
            VERIFY_EQUALS( 52, V1, V1, V3, V1, V2 )
            VERIFY_EQUALS( 53, V1, V1, V3, V2, V1 )
            VERIFY_EQUALS( 54, V1, V1, V4, V1, V1 )
            VERIFY_EQUALS( 55, V1, V2, V1, V1, V3 )
            VERIFY_EQUALS( 56, V1, V2, V1, V2, V2 )
            VERIFY_EQUALS( 57, V1, V2, V1, V3, V1 )
            VERIFY_EQUALS( 58, V1, V2, V2, V1, V2 )
            VERIFY_EQUALS( 59, V1, V2, V2, V2, V1 )
            VERIFY_EQUALS( 60, V1, V2, V3, V1, V1 )
            VERIFY_EQUALS( 61, V1, V3, V1, V1, V2 )
            VERIFY_EQUALS( 62, V1, V3, V1, V2, V1 )
            VERIFY_EQUALS( 63, V1, V3, V2, V1, V1 )
            VERIFY_EQUALS( 64, V1, V4, V1, V1, V1 )
            VERIFY_EQUALS( 65, V2, V1, V1, V1, V3 )
            VERIFY_EQUALS( 66, V2, V1, V1, V2, V2 )
            VERIFY_EQUALS( 67, V2, V1, V1, V3, V1 )
            VERIFY_EQUALS( 68, V2, V1, V2, V1, V2 )
            VERIFY_EQUALS( 69, V2, V1, V2, V2, V1 )
            VERIFY_EQUALS( 70, V2, V1, V3, V1, V1 )
            VERIFY_EQUALS( 71, V2, V2, V1, V1, V2 )
            VERIFY_EQUALS( 72, V2, V2, V1, V2, V1 )
            VERIFY_EQUALS( 73, V2, V2, V2, V1, V1 )
            VERIFY_EQUALS( 74, V2, V3, V1, V1, V1 )
            VERIFY_EQUALS( 75, V3, V1, V1, V1, V2 )
            VERIFY_EQUALS( 76, V3, V1, V1, V2, V1 )
            VERIFY_EQUALS( 77, V3, V1, V2, V1, V1 )
            VERIFY_EQUALS( 78, V3, V2, V1, V1, V1 )
            VERIFY_EQUALS( 79, V4, V1, V1, V1, V1 )
            VERIFY_EQUALS( 80, V1, V1, V1, V1, V1, V3 )
            VERIFY_EQUALS( 81, V1, V1, V1, V1, V2, V2 )
            VERIFY_EQUALS( 82, V1, V1, V1, V1, V3, V1 )
            VERIFY_EQUALS( 83, V1, V1, V1, V2, V1, V2 )
            VERIFY_EQUALS( 84, V1, V1, V1, V2, V2, V1 )
            VERIFY_EQUALS( 85, V1, V1, V1, V3, V1, V1 )
            VERIFY_EQUALS( 86, V1, V1, V2, V1, V1, V2 )
            VERIFY_EQUALS( 87, V1, V1, V2, V1, V2, V1 )
            VERIFY_EQUALS( 88, V1, V1, V2, V2, V1, V1 )
            VERIFY_EQUALS( 89, V1, V1, V3, V1, V1, V1 )
            VERIFY_EQUALS( 90, V1, V2, V1, V1, V1, V2 )
            VERIFY_EQUALS( 91, V1, V2, V1, V1, V2, V1 )
            VERIFY_EQUALS( 92, V1, V2, V1, V2, V1, V1 )
            VERIFY_EQUALS( 93, V1, V2, V2, V1, V1, V1 )
            VERIFY_EQUALS( 94, V1, V3, V1, V1, V1, V1 )
            VERIFY_EQUALS( 95, V2, V1, V1, V1, V1, V2 )
            VERIFY_EQUALS( 96, V2, V1, V1, V1, V2, V1 )
            VERIFY_EQUALS( 97, V2, V1, V1, V2, V1, V1 )
            VERIFY_EQUALS( 98, V2, V1, V2, V1, V1, V1 )
            VERIFY_EQUALS( 99, V2, V2, V1, V1, V1, V1 )
            VERIFY_EQUALS( 100, V3, V1, V1, V1, V1, V1 )
            VERIFY_EQUALS( 101, V1, V1, V1, V1, V1, V1, V2 )
            VERIFY_EQUALS( 102, V1, V1, V1, V1, V1, V2, V1 )
            VERIFY_EQUALS( 103, V1, V1, V1, V1, V2, V1, V1 )
            VERIFY_EQUALS( 104, V1, V1, V1, V2, V1, V1, V1 )
            VERIFY_EQUALS( 105, V1, V1, V2, V1, V1, V1, V1 )
            VERIFY_EQUALS( 106, V1, V2, V1, V1, V1, V1, V1 )
            VERIFY_EQUALS( 107, V2, V1, V1, V1, V1, V1, V1 )
            VERIFY_EQUALS( 108, V1, V1, V1, V1, V1, V1, V1, V1 )
            
            /* MACROS GENERATED FROM PYTHON SCRIPT*/

            l_queue.wait_and_throw();
       }
        catch (cl::sycl::exception e)
        {
            log_exception(log, e);
            FAIL(log, "sycl exception caught");
        }

    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME<char8>>  proxy4;
}; /* vector_initalization__ */
