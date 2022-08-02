/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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