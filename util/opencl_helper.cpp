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

#include "opencl_helper.h"

namespace sycl_cts {
namespace util {

/*  */
bool opencl_helper::check_cl_success(logger &log, const cl_int clError,
                                     const int line) {
  if (clError != CL_SUCCESS) {
    std::string err_msg("CL_SUCCESS expected, got ");
    err_msg.append(std::to_string(clError));
    log.fail(err_msg, line);
  }
  return clError == CL_SUCCESS;
}

}  // namespace util
}  // namespace sycl_cts
