/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "opencl_helper.h"

namespace sycl_cts
{
namespace util
{

/*  */
bool opencl_helper::check_cl_success( logger &log, const cl_int clError, const int line )
{
    if ( clError != CL_SUCCESS )
    {
        std::string err_msg( "CL_SUCCESS expected, got " );
        err_msg.append( std::to_string( clError ) );
        log.fail( err_msg, line );
    }
    return clError == CL_SUCCESS;
}

}  // namespace util
}  // namespace sycl_cts
