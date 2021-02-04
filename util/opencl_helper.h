/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_OPENCL_HELPER_H
#define __SYCLCTS_UTIL_OPENCL_HELPER_H

#include <CL/cl.h>

#include "singleton.h"
#include "logger.h"

namespace sycl_cts {
namespace util {

/* helper functions for OpenCL code
 */
class opencl_helper : public singleton<opencl_helper> {
 public:
  /* check for an opencl error */
  bool check_cl_success(logger &log, const cl_int clError, const int line);
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_OPENCL_HELPER_H