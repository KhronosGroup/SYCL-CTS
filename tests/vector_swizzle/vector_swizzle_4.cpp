
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_swizzle.py
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

#define TEST_NAME vector_swizzles_4
#define KERNEL_NAME cKernel_vector_swizzles
#define VECTOR_SIZE 4
#define NUM_TESTS 96

/* SWIZZLE TESTS SPECIAL MACROS FOR EASIER GENERATION AND VERIFICATION*/
#define X 0
#define Y 1
#define Z 2
#define W 3
#define CREATE_VECTOR_TYPE(TYPE, SIZE) TYPE##SIZE
#define RHS_SIMPLE_SWIZZLE(INDEX, VARIATION) m_o[INDEX] = m_i[INDEX].VARIATION();
#define LHS_SIMPLE_SWIZZLE(INDEX, VARIATION) m_o[INDEX].VARIATION() = m_i[INDEX];
#define RHS_TEMPLATE_SWIZZLE(INDEX, ...) m_o[INDEX] = m_i[INDEX].template swizzle< __VA_ARGS__ >();
#define LHS_TEMPLATE_SWIZZLE(INDEX, ...) m_o[INDEX].template swizzle< __VA_ARGS__ >() = m_i[INDEX];
#define SWIZZLE_VERIFY_EQUALS(INDEX, ...) if( all( odata[INDEX] != T(__VA_ARGS__) )  ) { FAIL(log, "results don't match"); }


namespace TEST_NAMESPACE
{
using namespace sycl_cts;

/** kernel functor
 */
template <typename T>
class KERNEL_NAME
{
protected:
    typedef accessor<T, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> t_readAccess;
    typedef accessor<T, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> t_writeAccess;

    t_writeAccess m_o; /* output     */
    t_readAccess m_i;  /* input */

public:
    KERNEL_NAME(t_writeAccess out_, t_readAccess in_)
        : m_o(out_)
        , m_i(in_)
    {
    }

    void operator()(item<1> item)
    {
        /* MACROS GENERATED FROM PYTHON SCRIPT*/
        
        RHS_SIMPLE_SWIZZLE( 0, xyzw )
        RHS_SIMPLE_SWIZZLE( 1, xywz )
        RHS_SIMPLE_SWIZZLE( 2, xzyw )
        RHS_SIMPLE_SWIZZLE( 3, xzwy )
        RHS_SIMPLE_SWIZZLE( 4, xwyz )
        RHS_SIMPLE_SWIZZLE( 5, xwzy )
        RHS_SIMPLE_SWIZZLE( 6, yxzw )
        RHS_SIMPLE_SWIZZLE( 7, yxwz )
        RHS_SIMPLE_SWIZZLE( 8, yzxw )
        RHS_SIMPLE_SWIZZLE( 9, yzwx )
        RHS_SIMPLE_SWIZZLE( 10, ywxz )
        RHS_SIMPLE_SWIZZLE( 11, ywzx )
        RHS_SIMPLE_SWIZZLE( 12, zxyw )
        RHS_SIMPLE_SWIZZLE( 13, zxwy )
        RHS_SIMPLE_SWIZZLE( 14, zyxw )
        RHS_SIMPLE_SWIZZLE( 15, zywx )
        RHS_SIMPLE_SWIZZLE( 16, zwxy )
        RHS_SIMPLE_SWIZZLE( 17, zwyx )
        RHS_SIMPLE_SWIZZLE( 18, wxyz )
        RHS_SIMPLE_SWIZZLE( 19, wxzy )
        RHS_SIMPLE_SWIZZLE( 20, wyxz )
        RHS_SIMPLE_SWIZZLE( 21, wyzx )
        RHS_SIMPLE_SWIZZLE( 22, wzxy )
        RHS_SIMPLE_SWIZZLE( 23, wzyx )
        
        
        LHS_SIMPLE_SWIZZLE( 24, xyzw )
        LHS_SIMPLE_SWIZZLE( 25, xywz )
        LHS_SIMPLE_SWIZZLE( 26, xzyw )
        LHS_SIMPLE_SWIZZLE( 27, xzwy )
        LHS_SIMPLE_SWIZZLE( 28, xwyz )
        LHS_SIMPLE_SWIZZLE( 29, xwzy )
        LHS_SIMPLE_SWIZZLE( 30, yxzw )
        LHS_SIMPLE_SWIZZLE( 31, yxwz )
        LHS_SIMPLE_SWIZZLE( 32, yzxw )
        LHS_SIMPLE_SWIZZLE( 33, yzwx )
        LHS_SIMPLE_SWIZZLE( 34, ywxz )
        LHS_SIMPLE_SWIZZLE( 35, ywzx )
        LHS_SIMPLE_SWIZZLE( 36, zxyw )
        LHS_SIMPLE_SWIZZLE( 37, zxwy )
        LHS_SIMPLE_SWIZZLE( 38, zyxw )
        LHS_SIMPLE_SWIZZLE( 39, zywx )
        LHS_SIMPLE_SWIZZLE( 40, zwxy )
        LHS_SIMPLE_SWIZZLE( 41, zwyx )
        LHS_SIMPLE_SWIZZLE( 42, wxyz )
        LHS_SIMPLE_SWIZZLE( 43, wxzy )
        LHS_SIMPLE_SWIZZLE( 44, wyxz )
        LHS_SIMPLE_SWIZZLE( 45, wyzx )
        LHS_SIMPLE_SWIZZLE( 46, wzxy )
        LHS_SIMPLE_SWIZZLE( 47, wzyx )
        
        
        RHS_TEMPLATE_SWIZZLE( 48, X, Y, Z, W )
        RHS_TEMPLATE_SWIZZLE( 49, X, Y, W, Z )
        RHS_TEMPLATE_SWIZZLE( 50, X, Z, Y, W )
        RHS_TEMPLATE_SWIZZLE( 51, X, Z, W, Y )
        RHS_TEMPLATE_SWIZZLE( 52, X, W, Y, Z )
        RHS_TEMPLATE_SWIZZLE( 53, X, W, Z, Y )
        RHS_TEMPLATE_SWIZZLE( 54, Y, X, Z, W )
        RHS_TEMPLATE_SWIZZLE( 55, Y, X, W, Z )
        RHS_TEMPLATE_SWIZZLE( 56, Y, Z, X, W )
        RHS_TEMPLATE_SWIZZLE( 57, Y, Z, W, X )
        RHS_TEMPLATE_SWIZZLE( 58, Y, W, X, Z )
        RHS_TEMPLATE_SWIZZLE( 59, Y, W, Z, X )
        RHS_TEMPLATE_SWIZZLE( 60, Z, X, Y, W )
        RHS_TEMPLATE_SWIZZLE( 61, Z, X, W, Y )
        RHS_TEMPLATE_SWIZZLE( 62, Z, Y, X, W )
        RHS_TEMPLATE_SWIZZLE( 63, Z, Y, W, X )
        RHS_TEMPLATE_SWIZZLE( 64, Z, W, X, Y )
        RHS_TEMPLATE_SWIZZLE( 65, Z, W, Y, X )
        RHS_TEMPLATE_SWIZZLE( 66, W, X, Y, Z )
        RHS_TEMPLATE_SWIZZLE( 67, W, X, Z, Y )
        RHS_TEMPLATE_SWIZZLE( 68, W, Y, X, Z )
        RHS_TEMPLATE_SWIZZLE( 69, W, Y, Z, X )
        RHS_TEMPLATE_SWIZZLE( 70, W, Z, X, Y )
        RHS_TEMPLATE_SWIZZLE( 71, W, Z, Y, X )
        
        
        LHS_TEMPLATE_SWIZZLE( 72, X, Y, Z, W )
        LHS_TEMPLATE_SWIZZLE( 73, X, Y, W, Z )
        LHS_TEMPLATE_SWIZZLE( 74, X, Z, Y, W )
        LHS_TEMPLATE_SWIZZLE( 75, X, Z, W, Y )
        LHS_TEMPLATE_SWIZZLE( 76, X, W, Y, Z )
        LHS_TEMPLATE_SWIZZLE( 77, X, W, Z, Y )
        LHS_TEMPLATE_SWIZZLE( 78, Y, X, Z, W )
        LHS_TEMPLATE_SWIZZLE( 79, Y, X, W, Z )
        LHS_TEMPLATE_SWIZZLE( 80, Y, Z, X, W )
        LHS_TEMPLATE_SWIZZLE( 81, Y, Z, W, X )
        LHS_TEMPLATE_SWIZZLE( 82, Y, W, X, Z )
        LHS_TEMPLATE_SWIZZLE( 83, Y, W, Z, X )
        LHS_TEMPLATE_SWIZZLE( 84, Z, X, Y, W )
        LHS_TEMPLATE_SWIZZLE( 85, Z, X, W, Y )
        LHS_TEMPLATE_SWIZZLE( 86, Z, Y, X, W )
        LHS_TEMPLATE_SWIZZLE( 87, Z, Y, W, X )
        LHS_TEMPLATE_SWIZZLE( 88, Z, W, X, Y )
        LHS_TEMPLATE_SWIZZLE( 89, Z, W, Y, X )
        LHS_TEMPLATE_SWIZZLE( 90, W, X, Y, Z )
        LHS_TEMPLATE_SWIZZLE( 91, W, X, Z, Y )
        LHS_TEMPLATE_SWIZZLE( 92, W, Y, X, Z )
        LHS_TEMPLATE_SWIZZLE( 93, W, Y, Z, X )
        LHS_TEMPLATE_SWIZZLE( 94, W, Z, X, Y )
        LHS_TEMPLATE_SWIZZLE( 95, W, Z, Y, X )
        
        /* MACROS GENERATED FROM PYTHON SCRIPT*/
    }
};

/** test SYCL header for compilation
 */
template <typename T, typename E>
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     */
    virtual void get_info( test_base::info &out ) const override
    {
        using sycl_cts::util::STRING;
        STRING name = STRING( TOSTRING( TEST_NAME ) + STRING( "_" ) + type_name<T>( ) );
        set_test_info( out, name.c_str( ), TEST_FILE );
    }

    /** execute this test
     */
    virtual void run(util::logger &log)
    {
        try
        {
            E vector_input_data[4] = {X, Y, Z, W};
            T idata[NUM_TESTS];
            T odata[NUM_TESTS];

            for (int i = 0; i < NUM_TESTS; i++)
            {
                for (int j = 0; j < VECTOR_SIZE; j++)
                {
                    idata[i][j] = vector_input_data[j];
                }
            }

            // construct the cts default selector
            cts_selector selector;

            /* create command queue */
            queue l_queue(selector);

            buffer<T, 1> ibuf(idata, range<1> ( NUM_TESTS ) );
            buffer<T, 1> obuf(odata, range<1> ( NUM_TESTS ) );

            /* add command to queue */
            l_queue.submit( [&]( handler& cgh )
            {
                auto iptr = ibuf.template get_access<access::read>( cgh );
                auto optr = obuf.template get_access<cl::sycl::access::mode::write>( cgh );

                /* instantiate the kernel */
                auto kern = KERNEL_NAME<T>( optr, iptr );

                /* execute the kernel */
                cgh.parallel_for( nd_range<1>( range<1>( 1 ), range<1>( 1 )), kern );
            } );

            /* MACROS GENERATED FROM PYTHON SCRIPT*/
            
            SWIZZLE_VERIFY_EQUALS( 0, X, Y, Z, W )
            SWIZZLE_VERIFY_EQUALS( 1, X, Y, W, Z )
            SWIZZLE_VERIFY_EQUALS( 2, X, Z, Y, W )
            SWIZZLE_VERIFY_EQUALS( 3, X, Z, W, Y )
            SWIZZLE_VERIFY_EQUALS( 4, X, W, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 5, X, W, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 6, Y, X, Z, W )
            SWIZZLE_VERIFY_EQUALS( 7, Y, X, W, Z )
            SWIZZLE_VERIFY_EQUALS( 8, Y, Z, X, W )
            SWIZZLE_VERIFY_EQUALS( 9, Y, Z, W, X )
            SWIZZLE_VERIFY_EQUALS( 10, Y, W, X, Z )
            SWIZZLE_VERIFY_EQUALS( 11, Y, W, Z, X )
            SWIZZLE_VERIFY_EQUALS( 12, Z, X, Y, W )
            SWIZZLE_VERIFY_EQUALS( 13, Z, X, W, Y )
            SWIZZLE_VERIFY_EQUALS( 14, Z, Y, X, W )
            SWIZZLE_VERIFY_EQUALS( 15, Z, Y, W, X )
            SWIZZLE_VERIFY_EQUALS( 16, Z, W, X, Y )
            SWIZZLE_VERIFY_EQUALS( 17, Z, W, Y, X )
            SWIZZLE_VERIFY_EQUALS( 18, W, X, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 19, W, X, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 20, W, Y, X, Z )
            SWIZZLE_VERIFY_EQUALS( 21, W, Y, Z, X )
            SWIZZLE_VERIFY_EQUALS( 22, W, Z, X, Y )
            SWIZZLE_VERIFY_EQUALS( 23, W, Z, Y, X )
            
            
            SWIZZLE_VERIFY_EQUALS( 24, X, Y, Z, W )
            SWIZZLE_VERIFY_EQUALS( 25, X, Y, W, Z )
            SWIZZLE_VERIFY_EQUALS( 26, X, Z, Y, W )
            SWIZZLE_VERIFY_EQUALS( 27, X, Z, W, Y )
            SWIZZLE_VERIFY_EQUALS( 28, X, W, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 29, X, W, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 30, Y, X, Z, W )
            SWIZZLE_VERIFY_EQUALS( 31, Y, X, W, Z )
            SWIZZLE_VERIFY_EQUALS( 32, Y, Z, X, W )
            SWIZZLE_VERIFY_EQUALS( 33, Y, Z, W, X )
            SWIZZLE_VERIFY_EQUALS( 34, Y, W, X, Z )
            SWIZZLE_VERIFY_EQUALS( 35, Y, W, Z, X )
            SWIZZLE_VERIFY_EQUALS( 36, Z, X, Y, W )
            SWIZZLE_VERIFY_EQUALS( 37, Z, X, W, Y )
            SWIZZLE_VERIFY_EQUALS( 38, Z, Y, X, W )
            SWIZZLE_VERIFY_EQUALS( 39, Z, Y, W, X )
            SWIZZLE_VERIFY_EQUALS( 40, Z, W, X, Y )
            SWIZZLE_VERIFY_EQUALS( 41, Z, W, Y, X )
            SWIZZLE_VERIFY_EQUALS( 42, W, X, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 43, W, X, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 44, W, Y, X, Z )
            SWIZZLE_VERIFY_EQUALS( 45, W, Y, Z, X )
            SWIZZLE_VERIFY_EQUALS( 46, W, Z, X, Y )
            SWIZZLE_VERIFY_EQUALS( 47, W, Z, Y, X )
            
            
            SWIZZLE_VERIFY_EQUALS( 48, X, Y, Z, W )
            SWIZZLE_VERIFY_EQUALS( 49, X, Y, W, Z )
            SWIZZLE_VERIFY_EQUALS( 50, X, Z, Y, W )
            SWIZZLE_VERIFY_EQUALS( 51, X, Z, W, Y )
            SWIZZLE_VERIFY_EQUALS( 52, X, W, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 53, X, W, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 54, Y, X, Z, W )
            SWIZZLE_VERIFY_EQUALS( 55, Y, X, W, Z )
            SWIZZLE_VERIFY_EQUALS( 56, Y, Z, X, W )
            SWIZZLE_VERIFY_EQUALS( 57, Y, Z, W, X )
            SWIZZLE_VERIFY_EQUALS( 58, Y, W, X, Z )
            SWIZZLE_VERIFY_EQUALS( 59, Y, W, Z, X )
            SWIZZLE_VERIFY_EQUALS( 60, Z, X, Y, W )
            SWIZZLE_VERIFY_EQUALS( 61, Z, X, W, Y )
            SWIZZLE_VERIFY_EQUALS( 62, Z, Y, X, W )
            SWIZZLE_VERIFY_EQUALS( 63, Z, Y, W, X )
            SWIZZLE_VERIFY_EQUALS( 64, Z, W, X, Y )
            SWIZZLE_VERIFY_EQUALS( 65, Z, W, Y, X )
            SWIZZLE_VERIFY_EQUALS( 66, W, X, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 67, W, X, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 68, W, Y, X, Z )
            SWIZZLE_VERIFY_EQUALS( 69, W, Y, Z, X )
            SWIZZLE_VERIFY_EQUALS( 70, W, Z, X, Y )
            SWIZZLE_VERIFY_EQUALS( 71, W, Z, Y, X )
            
            
            SWIZZLE_VERIFY_EQUALS( 72, X, Y, Z, W )
            SWIZZLE_VERIFY_EQUALS( 73, X, Y, W, Z )
            SWIZZLE_VERIFY_EQUALS( 74, X, Z, Y, W )
            SWIZZLE_VERIFY_EQUALS( 75, X, Z, W, Y )
            SWIZZLE_VERIFY_EQUALS( 76, X, W, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 77, X, W, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 78, Y, X, Z, W )
            SWIZZLE_VERIFY_EQUALS( 79, Y, X, W, Z )
            SWIZZLE_VERIFY_EQUALS( 80, Y, Z, X, W )
            SWIZZLE_VERIFY_EQUALS( 81, Y, Z, W, X )
            SWIZZLE_VERIFY_EQUALS( 82, Y, W, X, Z )
            SWIZZLE_VERIFY_EQUALS( 83, Y, W, Z, X )
            SWIZZLE_VERIFY_EQUALS( 84, Z, X, Y, W )
            SWIZZLE_VERIFY_EQUALS( 85, Z, X, W, Y )
            SWIZZLE_VERIFY_EQUALS( 86, Z, Y, X, W )
            SWIZZLE_VERIFY_EQUALS( 87, Z, Y, W, X )
            SWIZZLE_VERIFY_EQUALS( 88, Z, W, X, Y )
            SWIZZLE_VERIFY_EQUALS( 89, Z, W, Y, X )
            SWIZZLE_VERIFY_EQUALS( 90, W, X, Y, Z )
            SWIZZLE_VERIFY_EQUALS( 91, W, X, Z, Y )
            SWIZZLE_VERIFY_EQUALS( 92, W, Y, X, Z )
            SWIZZLE_VERIFY_EQUALS( 93, W, Y, Z, X )
            SWIZZLE_VERIFY_EQUALS( 94, W, Z, X, Y )
            SWIZZLE_VERIFY_EQUALS( 95, W, Z, Y, X )
            
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
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(char,  4),int8_t>>  proxy2;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(uchar, 4),uint8_t>>  proxy3;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(short, 4),int16_t>>  proxy4;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(ushort,4),uint16_t>>  proxy5;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(int,   4),int32_t>>  proxy6;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(uint,  4),uint32_t>>  proxy7;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(float, 4),float>>  proxy8;
util::test_proxy<TEST_NAME<CREATE_VECTOR_TYPE(double,4),double>>  proxy9;

}; /* vector_initalization__ */
