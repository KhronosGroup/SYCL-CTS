/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include <CL/sycl.hpp>

#define	TEST_FILE       __FILE__
#define	TEST_BUILD_DATE __DATE__
#define	TEST_BUILD_TIME __TIME__

/** convert a parameter to a string
 */
#define TOSTRING( X ) STRINGIFY( X )
#define STRINGIFY( X ) #X

/** helper function used to construct a typical test info
 *  structure
 */
static void set_test_info(
    sycl_cts::util::test_base::info & out,
    const char *name,
    const char *file )
{
    out.m_name      = name;
    out.m_file      = file;
    out.m_buildDate = TEST_BUILD_DATE;
    out.m_buildTime = TEST_BUILD_TIME;
}

/** 
 *
 */
static void log_exception(
    sycl_cts::util::logger & log,
    cl::sycl::sycl_error & e
    )
{
    // notify that an exception was thrown
    log.note( "sycl exception caught" );

#if 1

    // log exception error string
    const char *what = e.what( );
    if ( what != nullptr )
        log.note( "what - " + sycl_cts::util::STRING( what ) );

#else
    // this part may be specific to SYCLONE

    // log an opencl error message
    const char *cl_error_msg   = e.get_cl_error_message( );
    if ( cl_error_msg != nullptr )
        log.note( "cl_error - " + sycl_cts::util::STRING( cl_error_msg ) );

    // log a sycl error message
    const char *sycl_error_msg = e.get_sycl_error_message( );
    if ( sycl_error_msg != nullptr )
        log.note( "sycl_error - " + sycl_cts::util::STRING( sycl_error_msg ) );
#endif
}

/**
 * macro to record line numbers for failures
 */
#define FAIL( LOG, MSG ) ( LOG .fail( MSG, __LINE__ ) )
