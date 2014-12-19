/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "sycl.h"
#include "../../util/opencl_helper.h"

#define TEST_FILE __FILE__
#define TEST_BUILD_DATE __DATE__
#define TEST_BUILD_TIME __TIME__

/** convert a parameter to a string
 */
#define TOSTRING( X ) STRINGIFY( X )
#define STRINGIFY( X ) #X

namespace
{

/** helper function used to construct a typical test info
 *  structure
 */
void set_test_info( sycl_cts::util::test_base::info &out, const char *name, const char *file )
{
    out.m_name = name;
    out.m_file = file;
    out.m_buildDate = TEST_BUILD_DATE;
    out.m_buildTime = TEST_BUILD_TIME;
}

/**
 *
 */
void log_exception( sycl_cts::util::logger &log, cl::sycl::exception &e )
{
    // notify that an exception was thrown
    log.note( "sycl exception caught" );

#if 1
    // log exception error string
    const char *what = e.what();
    if ( what != nullptr )
        log.note( "what - " + sycl_cts::util::STRING( what ) );

#else
    // log an opencl error message
    const char *cl_error_msg = e.get_cl_error_message();
    if ( cl_error_msg != nullptr )
        log.note( "cl_error - " + sycl_cts::util::STRING( cl_error_msg ) );

    // log a sycl error message
    const char *sycl_error_msg = e.get_sycl_error_message();
    if ( sycl_error_msg != nullptr )
        log.note( "sycl_error - " + sycl_cts::util::STRING( sycl_error_msg ) );
		
#endif
}

/* helper function for test failure cases */
bool fail_proxy( sycl_cts::util::logger &log, const char *msg, int line )
{
    log.fail( msg, line );
    return false;
}

/* helper function for test failure cases */
bool fail_proxy( sycl_cts::util::logger &log, const sycl_cts::util::STRING & msg, int line )
{
    log.fail( msg, line );
    return false;
}

/* macro to record line numbers for failures */
#define FAIL( LOG, MSG ) ( fail_proxy( LOG, MSG, __LINE__ ) )

/* proxy to the check_cl_success function */
bool check_cl_success_proxy( sycl_cts::util::logger &log, int error, int line )
{
    using sycl_cts::util::get;
    using sycl_cts::util::opencl_helper;
    return get<opencl_helper>().check_cl_success( log, error, line );
}

/*  */
#define CHECK_CL_SUCCESS( LOG, ERROR ) check_cl_success_proxy( LOG, ERROR, __LINE__ )

/* macro to check if provided value is equal to expected value */
template<typename T>
bool check_value_proxy( sycl_cts::util::logger &log, T got, T expected, int element, int line )
{
    if ( got != expected )
    {
        sycl_cts::util::STRING msg = 
            "Expected "    + std::to_string( expected ) + 
            " but got "    + std::to_string( got ) + 
            " at element " + std::to_string( element  );
        fail_proxy( log, msg.c_str(), line );
        /* values are different */
        return false;
    }
    /* values are equal */
    return true;
}

#define CHECK_VALUE( LOG, GOT, EXPECTED, INDEX ) check_value_proxy( LOG, GOT, EXPECTED, INDEX, __LINE__ )

/* macro to silence unused variables warnings */
#define UNUSED(expr) do { (void)(expr); } while (0)

} /* namespace {} */
