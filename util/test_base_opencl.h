/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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

#ifndef __SYCLCTS_UTIL_TEST_BASE_OPENCL_H
#define __SYCLCTS_UTIL_TEST_BASE_OPENCL_H

#include <CL/cl.h>

#include "stl.h"
#include "test_base.h"

// conformance test suite namespace
namespace sycl_cts {
namespace util {

/** Common base class for OpenCL inter operation tests
 */
class test_base_opencl : public sycl_cts::util::test_base {
 public:
  /** ctor
   */
  test_base_opencl();

  /** virtual destructor
   */
  virtual ~test_base_opencl() {}

  /** create an OpenCL buffer
   */
  bool create_buffer(cl_mem &outBuffer, size_t size, logger &log);

  /** create an OpenCL image
   */
  bool create_image(cl_mem &outImage, const cl_image_format &format,
                    const cl_image_desc &desc, logger &log);

  /** return a valid OpenCL cl_context object
   */
  cl_context get_cl_context() { return m_cl_context; }

 protected:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  virtual void get_info(test_base::info &out) const = 0;

  /** called before this test is executed
   *  @param log for emitting test notes and results
   */
  virtual bool setup(logger &log);

  /** execute this test
   *  @param log for emitting test notes and results
   */
  virtual void run(logger &log) = 0;

  /** called after this test has executed
   */
  virtual void cleanup();

  /** return a valid OpenCL platform object
   */
  cl_platform_id get_cl_platform_id() { return m_cl_platform_id; }

  /** return a valid OpenCL cl_device_id object
   */
  cl_device_id get_cl_device_id() { return m_cl_device_id; }

  /** return a valid OpenCL cl_command_queue object
   */
  cl_command_queue get_cl_command_queue() { return m_cl_command_queue; }

  bool get_exec_dir(char *path, size_t max_path_len);

  bool online_compiler_supported(cl_device_id clDeviceId, logger &log);

  /** create and compile an OpenCL program
   */
  bool create_compiled_program(const std::string &source,
                               cl_program &outProgram, logger &log);
  bool create_compiled_program(const std::string &source, cl_context clContext,
                               cl_device_id clDeviceId, cl_program &outProgram,
                               logger &log);

  /** create and build an OpenCL program
   */
  bool create_built_program(const std::string &source, cl_program &outProgram,
                            logger &log);
  bool create_built_program(const std::string &source, cl_context clContext,
                            cl_device_id clDeviceId, cl_program &outProgram,
                            logger &log);

  /** create an OpenCL program with binary
   */
  bool create_program_with_binary(const std::string &filename,
                                  cl_program &outProgram, logger &log);
  bool create_program_with_binary(const std::string &filename,
                                  cl_context clContext, cl_device_id clDeviceId,
                                  cl_program &outProgram, logger &log);

  /** create an OpenCL kernel
   */
  bool create_kernel(const cl_program &clProgram, const std::string &name,
                     cl_kernel &outKernel, logger &log);

  /** create an OpenCL sampler
   */
  bool create_sampler(cl_sampler &outSampler, logger &log);

  /* instances of OpenCL objects */
  cl_platform_id m_cl_platform_id;
  cl_device_id m_cl_device_id;
  cl_context m_cl_context;
  cl_command_queue m_cl_command_queue;

  /*  */
  std::vector<cl_kernel> m_openKernels;
  std::vector<cl_program> m_openPrograms;
  std::vector<cl_mem> m_openBuffers;
  std::vector<cl_mem> m_openImages;

  std::vector<std::shared_ptr<uint8_t>> m_openBufferHostPtrs;

};  // class test_base

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_TEST_BASE_OPENCL_H
