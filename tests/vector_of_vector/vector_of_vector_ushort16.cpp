
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

#define TEST_NAME vector_of_vector_ushort16
#define KERNEL_NAME cKernel_vector_of_vector
#define NUM_TESTS 199

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

#define ushort1 ushort
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
        ushort1  v1  (V1S0);
    	ushort2  v2  (V2S0,  V2S1);
		ushort3  v3  (V3S0,  V3S1,  V3S2);
		ushort4  v4  (V4S0,  V4S1,  V4S2,  V4S3);
		ushort8  v8  (V8S0,  V8S1,  V8S2,  V8S3,  V8S4,  V8S5,  V8S6,  V8S7);
		ushort16 v16 (V16S0, V16S1, V16S2, V16S3, V16S4, V16S5, V16S6, V16S7, V16S8, V16S9, V16S10, V16S11, V16S12, V16S13, V16S14, V16S15);

        /* MACROS GENERATED FROM PYTHON SCRIPT*/
        
        CONSTRUCTOR_TEST( 0, v16 )
        CONSTRUCTOR_TEST( 1, v8, v8 )
        CONSTRUCTOR_TEST( 2, v4, v4, v8 )
        CONSTRUCTOR_TEST( 3, v4, v8, v4 )
        CONSTRUCTOR_TEST( 4, v8, v4, v4 )
        CONSTRUCTOR_TEST( 5, v2, v2, v4, v8 )
        CONSTRUCTOR_TEST( 6, v2, v2, v8, v4 )
        CONSTRUCTOR_TEST( 7, v2, v3, v3, v8 )
        CONSTRUCTOR_TEST( 8, v2, v3, v8, v3 )
        CONSTRUCTOR_TEST( 9, v2, v4, v2, v8 )
        CONSTRUCTOR_TEST( 10, v2, v4, v8, v2 )
        CONSTRUCTOR_TEST( 11, v2, v8, v2, v4 )
        CONSTRUCTOR_TEST( 12, v2, v8, v3, v3 )
        CONSTRUCTOR_TEST( 13, v2, v8, v4, v2 )
        CONSTRUCTOR_TEST( 14, v3, v2, v3, v8 )
        CONSTRUCTOR_TEST( 15, v3, v2, v8, v3 )
        CONSTRUCTOR_TEST( 16, v3, v3, v2, v8 )
        CONSTRUCTOR_TEST( 17, v3, v3, v8, v2 )
        CONSTRUCTOR_TEST( 18, v3, v8, v2, v3 )
        CONSTRUCTOR_TEST( 19, v3, v8, v3, v2 )
        CONSTRUCTOR_TEST( 20, v4, v2, v2, v8 )
        CONSTRUCTOR_TEST( 21, v4, v2, v8, v2 )
        CONSTRUCTOR_TEST( 22, v4, v4, v4, v4 )
        CONSTRUCTOR_TEST( 23, v4, v8, v2, v2 )
        CONSTRUCTOR_TEST( 24, v8, v2, v2, v4 )
        CONSTRUCTOR_TEST( 25, v8, v2, v3, v3 )
        CONSTRUCTOR_TEST( 26, v8, v2, v4, v2 )
        CONSTRUCTOR_TEST( 27, v8, v3, v2, v3 )
        CONSTRUCTOR_TEST( 28, v8, v3, v3, v2 )
        CONSTRUCTOR_TEST( 29, v8, v4, v2, v2 )
        CONSTRUCTOR_TEST( 30, v2, v2, v2, v2, v8 )
        CONSTRUCTOR_TEST( 31, v2, v2, v2, v8, v2 )
        CONSTRUCTOR_TEST( 32, v2, v2, v4, v4, v4 )
        CONSTRUCTOR_TEST( 33, v2, v2, v8, v2, v2 )
        CONSTRUCTOR_TEST( 34, v2, v3, v3, v4, v4 )
        CONSTRUCTOR_TEST( 35, v2, v3, v4, v3, v4 )
        CONSTRUCTOR_TEST( 36, v2, v3, v4, v4, v3 )
        CONSTRUCTOR_TEST( 37, v2, v4, v2, v4, v4 )
        CONSTRUCTOR_TEST( 38, v2, v4, v3, v3, v4 )
        CONSTRUCTOR_TEST( 39, v2, v4, v3, v4, v3 )
        CONSTRUCTOR_TEST( 40, v2, v4, v4, v2, v4 )
        CONSTRUCTOR_TEST( 41, v2, v4, v4, v3, v3 )
        CONSTRUCTOR_TEST( 42, v2, v4, v4, v4, v2 )
        CONSTRUCTOR_TEST( 43, v2, v8, v2, v2, v2 )
        CONSTRUCTOR_TEST( 44, v3, v2, v3, v4, v4 )
        CONSTRUCTOR_TEST( 45, v3, v2, v4, v3, v4 )
        CONSTRUCTOR_TEST( 46, v3, v2, v4, v4, v3 )
        CONSTRUCTOR_TEST( 47, v3, v3, v2, v4, v4 )
        CONSTRUCTOR_TEST( 48, v3, v3, v3, v3, v4 )
        CONSTRUCTOR_TEST( 49, v3, v3, v3, v4, v3 )
        CONSTRUCTOR_TEST( 50, v3, v3, v4, v2, v4 )
        CONSTRUCTOR_TEST( 51, v3, v3, v4, v3, v3 )
        CONSTRUCTOR_TEST( 52, v3, v3, v4, v4, v2 )
        CONSTRUCTOR_TEST( 53, v3, v4, v2, v3, v4 )
        CONSTRUCTOR_TEST( 54, v3, v4, v2, v4, v3 )
        CONSTRUCTOR_TEST( 55, v3, v4, v3, v2, v4 )
        CONSTRUCTOR_TEST( 56, v3, v4, v3, v3, v3 )
        CONSTRUCTOR_TEST( 57, v3, v4, v3, v4, v2 )
        CONSTRUCTOR_TEST( 58, v3, v4, v4, v2, v3 )
        CONSTRUCTOR_TEST( 59, v3, v4, v4, v3, v2 )
        CONSTRUCTOR_TEST( 60, v4, v2, v2, v4, v4 )
        CONSTRUCTOR_TEST( 61, v4, v2, v3, v3, v4 )
        CONSTRUCTOR_TEST( 62, v4, v2, v3, v4, v3 )
        CONSTRUCTOR_TEST( 63, v4, v2, v4, v2, v4 )
        CONSTRUCTOR_TEST( 64, v4, v2, v4, v3, v3 )
        CONSTRUCTOR_TEST( 65, v4, v2, v4, v4, v2 )
        CONSTRUCTOR_TEST( 66, v4, v3, v2, v3, v4 )
        CONSTRUCTOR_TEST( 67, v4, v3, v2, v4, v3 )
        CONSTRUCTOR_TEST( 68, v4, v3, v3, v2, v4 )
        CONSTRUCTOR_TEST( 69, v4, v3, v3, v3, v3 )
        CONSTRUCTOR_TEST( 70, v4, v3, v3, v4, v2 )
        CONSTRUCTOR_TEST( 71, v4, v3, v4, v2, v3 )
        CONSTRUCTOR_TEST( 72, v4, v3, v4, v3, v2 )
        CONSTRUCTOR_TEST( 73, v4, v4, v2, v2, v4 )
        CONSTRUCTOR_TEST( 74, v4, v4, v2, v3, v3 )
        CONSTRUCTOR_TEST( 75, v4, v4, v2, v4, v2 )
        CONSTRUCTOR_TEST( 76, v4, v4, v3, v2, v3 )
        CONSTRUCTOR_TEST( 77, v4, v4, v3, v3, v2 )
        CONSTRUCTOR_TEST( 78, v4, v4, v4, v2, v2 )
        CONSTRUCTOR_TEST( 79, v8, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 80, v2, v2, v2, v2, v4, v4 )
        CONSTRUCTOR_TEST( 81, v2, v2, v2, v3, v3, v4 )
        CONSTRUCTOR_TEST( 82, v2, v2, v2, v3, v4, v3 )
        CONSTRUCTOR_TEST( 83, v2, v2, v2, v4, v2, v4 )
        CONSTRUCTOR_TEST( 84, v2, v2, v2, v4, v3, v3 )
        CONSTRUCTOR_TEST( 85, v2, v2, v2, v4, v4, v2 )
        CONSTRUCTOR_TEST( 86, v2, v2, v3, v2, v3, v4 )
        CONSTRUCTOR_TEST( 87, v2, v2, v3, v2, v4, v3 )
        CONSTRUCTOR_TEST( 88, v2, v2, v3, v3, v2, v4 )
        CONSTRUCTOR_TEST( 89, v2, v2, v3, v3, v3, v3 )
        CONSTRUCTOR_TEST( 90, v2, v2, v3, v3, v4, v2 )
        CONSTRUCTOR_TEST( 91, v2, v2, v3, v4, v2, v3 )
        CONSTRUCTOR_TEST( 92, v2, v2, v3, v4, v3, v2 )
        CONSTRUCTOR_TEST( 93, v2, v2, v4, v2, v2, v4 )
        CONSTRUCTOR_TEST( 94, v2, v2, v4, v2, v3, v3 )
        CONSTRUCTOR_TEST( 95, v2, v2, v4, v2, v4, v2 )
        CONSTRUCTOR_TEST( 96, v2, v2, v4, v3, v2, v3 )
        CONSTRUCTOR_TEST( 97, v2, v2, v4, v3, v3, v2 )
        CONSTRUCTOR_TEST( 98, v2, v2, v4, v4, v2, v2 )
        CONSTRUCTOR_TEST( 99, v2, v3, v2, v2, v3, v4 )
        CONSTRUCTOR_TEST( 100, v2, v3, v2, v2, v4, v3 )
        CONSTRUCTOR_TEST( 101, v2, v3, v2, v3, v2, v4 )
        CONSTRUCTOR_TEST( 102, v2, v3, v2, v3, v3, v3 )
        CONSTRUCTOR_TEST( 103, v2, v3, v2, v3, v4, v2 )
        CONSTRUCTOR_TEST( 104, v2, v3, v2, v4, v2, v3 )
        CONSTRUCTOR_TEST( 105, v2, v3, v2, v4, v3, v2 )
        CONSTRUCTOR_TEST( 106, v2, v3, v3, v2, v2, v4 )
        CONSTRUCTOR_TEST( 107, v2, v3, v3, v2, v3, v3 )
        CONSTRUCTOR_TEST( 108, v2, v3, v3, v2, v4, v2 )
        CONSTRUCTOR_TEST( 109, v2, v3, v3, v3, v2, v3 )
        CONSTRUCTOR_TEST( 110, v2, v3, v3, v3, v3, v2 )
        CONSTRUCTOR_TEST( 111, v2, v3, v3, v4, v2, v2 )
        CONSTRUCTOR_TEST( 112, v2, v3, v4, v2, v2, v3 )
        CONSTRUCTOR_TEST( 113, v2, v3, v4, v2, v3, v2 )
        CONSTRUCTOR_TEST( 114, v2, v3, v4, v3, v2, v2 )
        CONSTRUCTOR_TEST( 115, v2, v4, v2, v2, v2, v4 )
        CONSTRUCTOR_TEST( 116, v2, v4, v2, v2, v3, v3 )
        CONSTRUCTOR_TEST( 117, v2, v4, v2, v2, v4, v2 )
        CONSTRUCTOR_TEST( 118, v2, v4, v2, v3, v2, v3 )
        CONSTRUCTOR_TEST( 119, v2, v4, v2, v3, v3, v2 )
        CONSTRUCTOR_TEST( 120, v2, v4, v2, v4, v2, v2 )
        CONSTRUCTOR_TEST( 121, v2, v4, v3, v2, v2, v3 )
        CONSTRUCTOR_TEST( 122, v2, v4, v3, v2, v3, v2 )
        CONSTRUCTOR_TEST( 123, v2, v4, v3, v3, v2, v2 )
        CONSTRUCTOR_TEST( 124, v2, v4, v4, v2, v2, v2 )
        CONSTRUCTOR_TEST( 125, v3, v2, v2, v2, v3, v4 )
        CONSTRUCTOR_TEST( 126, v3, v2, v2, v2, v4, v3 )
        CONSTRUCTOR_TEST( 127, v3, v2, v2, v3, v2, v4 )
        CONSTRUCTOR_TEST( 128, v3, v2, v2, v3, v3, v3 )
        CONSTRUCTOR_TEST( 129, v3, v2, v2, v3, v4, v2 )
        CONSTRUCTOR_TEST( 130, v3, v2, v2, v4, v2, v3 )
        CONSTRUCTOR_TEST( 131, v3, v2, v2, v4, v3, v2 )
        CONSTRUCTOR_TEST( 132, v3, v2, v3, v2, v2, v4 )
        CONSTRUCTOR_TEST( 133, v3, v2, v3, v2, v3, v3 )
        CONSTRUCTOR_TEST( 134, v3, v2, v3, v2, v4, v2 )
        CONSTRUCTOR_TEST( 135, v3, v2, v3, v3, v2, v3 )
        CONSTRUCTOR_TEST( 136, v3, v2, v3, v3, v3, v2 )
        CONSTRUCTOR_TEST( 137, v3, v2, v3, v4, v2, v2 )
        CONSTRUCTOR_TEST( 138, v3, v2, v4, v2, v2, v3 )
        CONSTRUCTOR_TEST( 139, v3, v2, v4, v2, v3, v2 )
        CONSTRUCTOR_TEST( 140, v3, v2, v4, v3, v2, v2 )
        CONSTRUCTOR_TEST( 141, v3, v3, v2, v2, v2, v4 )
        CONSTRUCTOR_TEST( 142, v3, v3, v2, v2, v3, v3 )
        CONSTRUCTOR_TEST( 143, v3, v3, v2, v2, v4, v2 )
        CONSTRUCTOR_TEST( 144, v3, v3, v2, v3, v2, v3 )
        CONSTRUCTOR_TEST( 145, v3, v3, v2, v3, v3, v2 )
        CONSTRUCTOR_TEST( 146, v3, v3, v2, v4, v2, v2 )
        CONSTRUCTOR_TEST( 147, v3, v3, v3, v2, v2, v3 )
        CONSTRUCTOR_TEST( 148, v3, v3, v3, v2, v3, v2 )
        CONSTRUCTOR_TEST( 149, v3, v3, v3, v3, v2, v2 )
        CONSTRUCTOR_TEST( 150, v3, v3, v4, v2, v2, v2 )
        CONSTRUCTOR_TEST( 151, v3, v4, v2, v2, v2, v3 )
        CONSTRUCTOR_TEST( 152, v3, v4, v2, v2, v3, v2 )
        CONSTRUCTOR_TEST( 153, v3, v4, v2, v3, v2, v2 )
        CONSTRUCTOR_TEST( 154, v3, v4, v3, v2, v2, v2 )
        CONSTRUCTOR_TEST( 155, v4, v2, v2, v2, v2, v4 )
        CONSTRUCTOR_TEST( 156, v4, v2, v2, v2, v3, v3 )
        CONSTRUCTOR_TEST( 157, v4, v2, v2, v2, v4, v2 )
        CONSTRUCTOR_TEST( 158, v4, v2, v2, v3, v2, v3 )
        CONSTRUCTOR_TEST( 159, v4, v2, v2, v3, v3, v2 )
        CONSTRUCTOR_TEST( 160, v4, v2, v2, v4, v2, v2 )
        CONSTRUCTOR_TEST( 161, v4, v2, v3, v2, v2, v3 )
        CONSTRUCTOR_TEST( 162, v4, v2, v3, v2, v3, v2 )
        CONSTRUCTOR_TEST( 163, v4, v2, v3, v3, v2, v2 )
        CONSTRUCTOR_TEST( 164, v4, v2, v4, v2, v2, v2 )
        CONSTRUCTOR_TEST( 165, v4, v3, v2, v2, v2, v3 )
        CONSTRUCTOR_TEST( 166, v4, v3, v2, v2, v3, v2 )
        CONSTRUCTOR_TEST( 167, v4, v3, v2, v3, v2, v2 )
        CONSTRUCTOR_TEST( 168, v4, v3, v3, v2, v2, v2 )
        CONSTRUCTOR_TEST( 169, v4, v4, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 170, v2, v2, v2, v2, v2, v2, v4 )
        CONSTRUCTOR_TEST( 171, v2, v2, v2, v2, v2, v3, v3 )
        CONSTRUCTOR_TEST( 172, v2, v2, v2, v2, v2, v4, v2 )
        CONSTRUCTOR_TEST( 173, v2, v2, v2, v2, v3, v2, v3 )
        CONSTRUCTOR_TEST( 174, v2, v2, v2, v2, v3, v3, v2 )
        CONSTRUCTOR_TEST( 175, v2, v2, v2, v2, v4, v2, v2 )
        CONSTRUCTOR_TEST( 176, v2, v2, v2, v3, v2, v2, v3 )
        CONSTRUCTOR_TEST( 177, v2, v2, v2, v3, v2, v3, v2 )
        CONSTRUCTOR_TEST( 178, v2, v2, v2, v3, v3, v2, v2 )
        CONSTRUCTOR_TEST( 179, v2, v2, v2, v4, v2, v2, v2 )
        CONSTRUCTOR_TEST( 180, v2, v2, v3, v2, v2, v2, v3 )
        CONSTRUCTOR_TEST( 181, v2, v2, v3, v2, v2, v3, v2 )
        CONSTRUCTOR_TEST( 182, v2, v2, v3, v2, v3, v2, v2 )
        CONSTRUCTOR_TEST( 183, v2, v2, v3, v3, v2, v2, v2 )
        CONSTRUCTOR_TEST( 184, v2, v2, v4, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 185, v2, v3, v2, v2, v2, v2, v3 )
        CONSTRUCTOR_TEST( 186, v2, v3, v2, v2, v2, v3, v2 )
        CONSTRUCTOR_TEST( 187, v2, v3, v2, v2, v3, v2, v2 )
        CONSTRUCTOR_TEST( 188, v2, v3, v2, v3, v2, v2, v2 )
        CONSTRUCTOR_TEST( 189, v2, v3, v3, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 190, v2, v4, v2, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 191, v3, v2, v2, v2, v2, v2, v3 )
        CONSTRUCTOR_TEST( 192, v3, v2, v2, v2, v2, v3, v2 )
        CONSTRUCTOR_TEST( 193, v3, v2, v2, v2, v3, v2, v2 )
        CONSTRUCTOR_TEST( 194, v3, v2, v2, v3, v2, v2, v2 )
        CONSTRUCTOR_TEST( 195, v3, v2, v3, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 196, v3, v3, v2, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 197, v4, v2, v2, v2, v2, v2, v2 )
        CONSTRUCTOR_TEST( 198, v2, v2, v2, v2, v2, v2, v2, v2 )
        
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
        MAKENAME(ushort2);
        MAKENAME(ushort3);
        MAKENAME(ushort4);
        MAKENAME(ushort8);
        MAKENAME(ushort16);
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
            
            VERIFY_EQUALS( 0, V16 )
            VERIFY_EQUALS( 1, V8, V8 )
            VERIFY_EQUALS( 2, V4, V4, V8 )
            VERIFY_EQUALS( 3, V4, V8, V4 )
            VERIFY_EQUALS( 4, V8, V4, V4 )
            VERIFY_EQUALS( 5, V2, V2, V4, V8 )
            VERIFY_EQUALS( 6, V2, V2, V8, V4 )
            VERIFY_EQUALS( 7, V2, V3, V3, V8 )
            VERIFY_EQUALS( 8, V2, V3, V8, V3 )
            VERIFY_EQUALS( 9, V2, V4, V2, V8 )
            VERIFY_EQUALS( 10, V2, V4, V8, V2 )
            VERIFY_EQUALS( 11, V2, V8, V2, V4 )
            VERIFY_EQUALS( 12, V2, V8, V3, V3 )
            VERIFY_EQUALS( 13, V2, V8, V4, V2 )
            VERIFY_EQUALS( 14, V3, V2, V3, V8 )
            VERIFY_EQUALS( 15, V3, V2, V8, V3 )
            VERIFY_EQUALS( 16, V3, V3, V2, V8 )
            VERIFY_EQUALS( 17, V3, V3, V8, V2 )
            VERIFY_EQUALS( 18, V3, V8, V2, V3 )
            VERIFY_EQUALS( 19, V3, V8, V3, V2 )
            VERIFY_EQUALS( 20, V4, V2, V2, V8 )
            VERIFY_EQUALS( 21, V4, V2, V8, V2 )
            VERIFY_EQUALS( 22, V4, V4, V4, V4 )
            VERIFY_EQUALS( 23, V4, V8, V2, V2 )
            VERIFY_EQUALS( 24, V8, V2, V2, V4 )
            VERIFY_EQUALS( 25, V8, V2, V3, V3 )
            VERIFY_EQUALS( 26, V8, V2, V4, V2 )
            VERIFY_EQUALS( 27, V8, V3, V2, V3 )
            VERIFY_EQUALS( 28, V8, V3, V3, V2 )
            VERIFY_EQUALS( 29, V8, V4, V2, V2 )
            VERIFY_EQUALS( 30, V2, V2, V2, V2, V8 )
            VERIFY_EQUALS( 31, V2, V2, V2, V8, V2 )
            VERIFY_EQUALS( 32, V2, V2, V4, V4, V4 )
            VERIFY_EQUALS( 33, V2, V2, V8, V2, V2 )
            VERIFY_EQUALS( 34, V2, V3, V3, V4, V4 )
            VERIFY_EQUALS( 35, V2, V3, V4, V3, V4 )
            VERIFY_EQUALS( 36, V2, V3, V4, V4, V3 )
            VERIFY_EQUALS( 37, V2, V4, V2, V4, V4 )
            VERIFY_EQUALS( 38, V2, V4, V3, V3, V4 )
            VERIFY_EQUALS( 39, V2, V4, V3, V4, V3 )
            VERIFY_EQUALS( 40, V2, V4, V4, V2, V4 )
            VERIFY_EQUALS( 41, V2, V4, V4, V3, V3 )
            VERIFY_EQUALS( 42, V2, V4, V4, V4, V2 )
            VERIFY_EQUALS( 43, V2, V8, V2, V2, V2 )
            VERIFY_EQUALS( 44, V3, V2, V3, V4, V4 )
            VERIFY_EQUALS( 45, V3, V2, V4, V3, V4 )
            VERIFY_EQUALS( 46, V3, V2, V4, V4, V3 )
            VERIFY_EQUALS( 47, V3, V3, V2, V4, V4 )
            VERIFY_EQUALS( 48, V3, V3, V3, V3, V4 )
            VERIFY_EQUALS( 49, V3, V3, V3, V4, V3 )
            VERIFY_EQUALS( 50, V3, V3, V4, V2, V4 )
            VERIFY_EQUALS( 51, V3, V3, V4, V3, V3 )
            VERIFY_EQUALS( 52, V3, V3, V4, V4, V2 )
            VERIFY_EQUALS( 53, V3, V4, V2, V3, V4 )
            VERIFY_EQUALS( 54, V3, V4, V2, V4, V3 )
            VERIFY_EQUALS( 55, V3, V4, V3, V2, V4 )
            VERIFY_EQUALS( 56, V3, V4, V3, V3, V3 )
            VERIFY_EQUALS( 57, V3, V4, V3, V4, V2 )
            VERIFY_EQUALS( 58, V3, V4, V4, V2, V3 )
            VERIFY_EQUALS( 59, V3, V4, V4, V3, V2 )
            VERIFY_EQUALS( 60, V4, V2, V2, V4, V4 )
            VERIFY_EQUALS( 61, V4, V2, V3, V3, V4 )
            VERIFY_EQUALS( 62, V4, V2, V3, V4, V3 )
            VERIFY_EQUALS( 63, V4, V2, V4, V2, V4 )
            VERIFY_EQUALS( 64, V4, V2, V4, V3, V3 )
            VERIFY_EQUALS( 65, V4, V2, V4, V4, V2 )
            VERIFY_EQUALS( 66, V4, V3, V2, V3, V4 )
            VERIFY_EQUALS( 67, V4, V3, V2, V4, V3 )
            VERIFY_EQUALS( 68, V4, V3, V3, V2, V4 )
            VERIFY_EQUALS( 69, V4, V3, V3, V3, V3 )
            VERIFY_EQUALS( 70, V4, V3, V3, V4, V2 )
            VERIFY_EQUALS( 71, V4, V3, V4, V2, V3 )
            VERIFY_EQUALS( 72, V4, V3, V4, V3, V2 )
            VERIFY_EQUALS( 73, V4, V4, V2, V2, V4 )
            VERIFY_EQUALS( 74, V4, V4, V2, V3, V3 )
            VERIFY_EQUALS( 75, V4, V4, V2, V4, V2 )
            VERIFY_EQUALS( 76, V4, V4, V3, V2, V3 )
            VERIFY_EQUALS( 77, V4, V4, V3, V3, V2 )
            VERIFY_EQUALS( 78, V4, V4, V4, V2, V2 )
            VERIFY_EQUALS( 79, V8, V2, V2, V2, V2 )
            VERIFY_EQUALS( 80, V2, V2, V2, V2, V4, V4 )
            VERIFY_EQUALS( 81, V2, V2, V2, V3, V3, V4 )
            VERIFY_EQUALS( 82, V2, V2, V2, V3, V4, V3 )
            VERIFY_EQUALS( 83, V2, V2, V2, V4, V2, V4 )
            VERIFY_EQUALS( 84, V2, V2, V2, V4, V3, V3 )
            VERIFY_EQUALS( 85, V2, V2, V2, V4, V4, V2 )
            VERIFY_EQUALS( 86, V2, V2, V3, V2, V3, V4 )
            VERIFY_EQUALS( 87, V2, V2, V3, V2, V4, V3 )
            VERIFY_EQUALS( 88, V2, V2, V3, V3, V2, V4 )
            VERIFY_EQUALS( 89, V2, V2, V3, V3, V3, V3 )
            VERIFY_EQUALS( 90, V2, V2, V3, V3, V4, V2 )
            VERIFY_EQUALS( 91, V2, V2, V3, V4, V2, V3 )
            VERIFY_EQUALS( 92, V2, V2, V3, V4, V3, V2 )
            VERIFY_EQUALS( 93, V2, V2, V4, V2, V2, V4 )
            VERIFY_EQUALS( 94, V2, V2, V4, V2, V3, V3 )
            VERIFY_EQUALS( 95, V2, V2, V4, V2, V4, V2 )
            VERIFY_EQUALS( 96, V2, V2, V4, V3, V2, V3 )
            VERIFY_EQUALS( 97, V2, V2, V4, V3, V3, V2 )
            VERIFY_EQUALS( 98, V2, V2, V4, V4, V2, V2 )
            VERIFY_EQUALS( 99, V2, V3, V2, V2, V3, V4 )
            VERIFY_EQUALS( 100, V2, V3, V2, V2, V4, V3 )
            VERIFY_EQUALS( 101, V2, V3, V2, V3, V2, V4 )
            VERIFY_EQUALS( 102, V2, V3, V2, V3, V3, V3 )
            VERIFY_EQUALS( 103, V2, V3, V2, V3, V4, V2 )
            VERIFY_EQUALS( 104, V2, V3, V2, V4, V2, V3 )
            VERIFY_EQUALS( 105, V2, V3, V2, V4, V3, V2 )
            VERIFY_EQUALS( 106, V2, V3, V3, V2, V2, V4 )
            VERIFY_EQUALS( 107, V2, V3, V3, V2, V3, V3 )
            VERIFY_EQUALS( 108, V2, V3, V3, V2, V4, V2 )
            VERIFY_EQUALS( 109, V2, V3, V3, V3, V2, V3 )
            VERIFY_EQUALS( 110, V2, V3, V3, V3, V3, V2 )
            VERIFY_EQUALS( 111, V2, V3, V3, V4, V2, V2 )
            VERIFY_EQUALS( 112, V2, V3, V4, V2, V2, V3 )
            VERIFY_EQUALS( 113, V2, V3, V4, V2, V3, V2 )
            VERIFY_EQUALS( 114, V2, V3, V4, V3, V2, V2 )
            VERIFY_EQUALS( 115, V2, V4, V2, V2, V2, V4 )
            VERIFY_EQUALS( 116, V2, V4, V2, V2, V3, V3 )
            VERIFY_EQUALS( 117, V2, V4, V2, V2, V4, V2 )
            VERIFY_EQUALS( 118, V2, V4, V2, V3, V2, V3 )
            VERIFY_EQUALS( 119, V2, V4, V2, V3, V3, V2 )
            VERIFY_EQUALS( 120, V2, V4, V2, V4, V2, V2 )
            VERIFY_EQUALS( 121, V2, V4, V3, V2, V2, V3 )
            VERIFY_EQUALS( 122, V2, V4, V3, V2, V3, V2 )
            VERIFY_EQUALS( 123, V2, V4, V3, V3, V2, V2 )
            VERIFY_EQUALS( 124, V2, V4, V4, V2, V2, V2 )
            VERIFY_EQUALS( 125, V3, V2, V2, V2, V3, V4 )
            VERIFY_EQUALS( 126, V3, V2, V2, V2, V4, V3 )
            VERIFY_EQUALS( 127, V3, V2, V2, V3, V2, V4 )
            VERIFY_EQUALS( 128, V3, V2, V2, V3, V3, V3 )
            VERIFY_EQUALS( 129, V3, V2, V2, V3, V4, V2 )
            VERIFY_EQUALS( 130, V3, V2, V2, V4, V2, V3 )
            VERIFY_EQUALS( 131, V3, V2, V2, V4, V3, V2 )
            VERIFY_EQUALS( 132, V3, V2, V3, V2, V2, V4 )
            VERIFY_EQUALS( 133, V3, V2, V3, V2, V3, V3 )
            VERIFY_EQUALS( 134, V3, V2, V3, V2, V4, V2 )
            VERIFY_EQUALS( 135, V3, V2, V3, V3, V2, V3 )
            VERIFY_EQUALS( 136, V3, V2, V3, V3, V3, V2 )
            VERIFY_EQUALS( 137, V3, V2, V3, V4, V2, V2 )
            VERIFY_EQUALS( 138, V3, V2, V4, V2, V2, V3 )
            VERIFY_EQUALS( 139, V3, V2, V4, V2, V3, V2 )
            VERIFY_EQUALS( 140, V3, V2, V4, V3, V2, V2 )
            VERIFY_EQUALS( 141, V3, V3, V2, V2, V2, V4 )
            VERIFY_EQUALS( 142, V3, V3, V2, V2, V3, V3 )
            VERIFY_EQUALS( 143, V3, V3, V2, V2, V4, V2 )
            VERIFY_EQUALS( 144, V3, V3, V2, V3, V2, V3 )
            VERIFY_EQUALS( 145, V3, V3, V2, V3, V3, V2 )
            VERIFY_EQUALS( 146, V3, V3, V2, V4, V2, V2 )
            VERIFY_EQUALS( 147, V3, V3, V3, V2, V2, V3 )
            VERIFY_EQUALS( 148, V3, V3, V3, V2, V3, V2 )
            VERIFY_EQUALS( 149, V3, V3, V3, V3, V2, V2 )
            VERIFY_EQUALS( 150, V3, V3, V4, V2, V2, V2 )
            VERIFY_EQUALS( 151, V3, V4, V2, V2, V2, V3 )
            VERIFY_EQUALS( 152, V3, V4, V2, V2, V3, V2 )
            VERIFY_EQUALS( 153, V3, V4, V2, V3, V2, V2 )
            VERIFY_EQUALS( 154, V3, V4, V3, V2, V2, V2 )
            VERIFY_EQUALS( 155, V4, V2, V2, V2, V2, V4 )
            VERIFY_EQUALS( 156, V4, V2, V2, V2, V3, V3 )
            VERIFY_EQUALS( 157, V4, V2, V2, V2, V4, V2 )
            VERIFY_EQUALS( 158, V4, V2, V2, V3, V2, V3 )
            VERIFY_EQUALS( 159, V4, V2, V2, V3, V3, V2 )
            VERIFY_EQUALS( 160, V4, V2, V2, V4, V2, V2 )
            VERIFY_EQUALS( 161, V4, V2, V3, V2, V2, V3 )
            VERIFY_EQUALS( 162, V4, V2, V3, V2, V3, V2 )
            VERIFY_EQUALS( 163, V4, V2, V3, V3, V2, V2 )
            VERIFY_EQUALS( 164, V4, V2, V4, V2, V2, V2 )
            VERIFY_EQUALS( 165, V4, V3, V2, V2, V2, V3 )
            VERIFY_EQUALS( 166, V4, V3, V2, V2, V3, V2 )
            VERIFY_EQUALS( 167, V4, V3, V2, V3, V2, V2 )
            VERIFY_EQUALS( 168, V4, V3, V3, V2, V2, V2 )
            VERIFY_EQUALS( 169, V4, V4, V2, V2, V2, V2 )
            VERIFY_EQUALS( 170, V2, V2, V2, V2, V2, V2, V4 )
            VERIFY_EQUALS( 171, V2, V2, V2, V2, V2, V3, V3 )
            VERIFY_EQUALS( 172, V2, V2, V2, V2, V2, V4, V2 )
            VERIFY_EQUALS( 173, V2, V2, V2, V2, V3, V2, V3 )
            VERIFY_EQUALS( 174, V2, V2, V2, V2, V3, V3, V2 )
            VERIFY_EQUALS( 175, V2, V2, V2, V2, V4, V2, V2 )
            VERIFY_EQUALS( 176, V2, V2, V2, V3, V2, V2, V3 )
            VERIFY_EQUALS( 177, V2, V2, V2, V3, V2, V3, V2 )
            VERIFY_EQUALS( 178, V2, V2, V2, V3, V3, V2, V2 )
            VERIFY_EQUALS( 179, V2, V2, V2, V4, V2, V2, V2 )
            VERIFY_EQUALS( 180, V2, V2, V3, V2, V2, V2, V3 )
            VERIFY_EQUALS( 181, V2, V2, V3, V2, V2, V3, V2 )
            VERIFY_EQUALS( 182, V2, V2, V3, V2, V3, V2, V2 )
            VERIFY_EQUALS( 183, V2, V2, V3, V3, V2, V2, V2 )
            VERIFY_EQUALS( 184, V2, V2, V4, V2, V2, V2, V2 )
            VERIFY_EQUALS( 185, V2, V3, V2, V2, V2, V2, V3 )
            VERIFY_EQUALS( 186, V2, V3, V2, V2, V2, V3, V2 )
            VERIFY_EQUALS( 187, V2, V3, V2, V2, V3, V2, V2 )
            VERIFY_EQUALS( 188, V2, V3, V2, V3, V2, V2, V2 )
            VERIFY_EQUALS( 189, V2, V3, V3, V2, V2, V2, V2 )
            VERIFY_EQUALS( 190, V2, V4, V2, V2, V2, V2, V2 )
            VERIFY_EQUALS( 191, V3, V2, V2, V2, V2, V2, V3 )
            VERIFY_EQUALS( 192, V3, V2, V2, V2, V2, V3, V2 )
            VERIFY_EQUALS( 193, V3, V2, V2, V2, V3, V2, V2 )
            VERIFY_EQUALS( 194, V3, V2, V2, V3, V2, V2, V2 )
            VERIFY_EQUALS( 195, V3, V2, V3, V2, V2, V2, V2 )
            VERIFY_EQUALS( 196, V3, V3, V2, V2, V2, V2, V2 )
            VERIFY_EQUALS( 197, V4, V2, V2, V2, V2, V2, V2 )
            VERIFY_EQUALS( 198, V2, V2, V2, V2, V2, V2, V2, V2 )
            
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
util::test_proxy<TEST_NAME<ushort16>>  proxy4;
}; /* vector_initalization__ */
