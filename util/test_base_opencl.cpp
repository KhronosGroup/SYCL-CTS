/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "test_base_opencl.h"
#include "../tests/common/macros.h"
#include "../util/opencl_helper.h"

// conformance test suite namespace
namespace sycl_cts {
namespace util {

/** constructor which explicitly sets the OpenCL objects
 *  to nullptrs
 */
test_base_opencl::test_base_opencl()
    : m_cl_platform_id(nullptr),
      m_cl_device_id(nullptr),
      m_cl_context(nullptr),
      m_cl_command_queue(nullptr),
      m_openKernels(),
      m_openPrograms() {}

/** called before this test is executed
 *  @param log for emitting test notes and results
 */
bool test_base_opencl::setup(logger &log) {
  /* get the OpenCLHelper object */
  using sycl_cts::util::opencl_helper;
  using sycl_cts::util::get;
  opencl_helper &openclHelper = get<opencl_helper>();

  cl::sycl::unique_ptr_class<cl_platform_id> platforms;
  cl::sycl::unique_ptr_class<cl_device_id> devices;

  cl_uint N = 0;
  cl_int error = CL_SUCCESS;

  error = clGetPlatformIDs(0, nullptr, &N);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  if (N < 1) {
    FAIL(log, "Unable to retrieve list of platforms via clGetPlatformIDs()");
    return false;
  }

  platforms.reset(new cl_platform_id[N]);
  if (platforms.get() == nullptr) {
    FAIL(log, "Malloc failed to allocate memory for platform list");
    return false;
  }

  error = clGetPlatformIDs(N, platforms.get(), nullptr);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  error =
      clGetDeviceIDs(platforms.get()[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &N);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  if (N < 1) {
    FAIL(log, "Unable to retrieve list of devices via clGetDeviceIDs()");
    return false;
  }

  devices.reset(new cl_device_id[N]);
  if (devices.get() == nullptr) {
    FAIL(log, "Malloc failed to allocate memory for device list");
    return false;
  }

  error = clGetDeviceIDs(platforms.get()[0], CL_DEVICE_TYPE_ALL, N,
                         devices.get(), nullptr);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platforms.get()[0], 0};

  m_cl_context =
      clCreateContext(properties, N, devices.get(), nullptr, nullptr, &error);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  // No special queue properties wanted
  m_cl_command_queue =
      clCreateCommandQueue(m_cl_context, devices.get()[0], 0, &error);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  m_cl_platform_id = platforms.get()[0];
  m_cl_device_id = devices.get()[0];

  return true;
}

bool test_base_opencl::create_compiled_program(const std::string &source,
                                               cl_program &out_program,
                                               logger &log) {
  assert(!source.empty());

  const char *source_c = source.c_str();
  const size_t sourceSize = source.length();
  assert(source_c != nullptr);

  cl_int error = CL_SUCCESS;
  out_program = clCreateProgramWithSource(m_cl_context, 1, &source_c,
                                          &sourceSize, &error);
  if (!CHECK_CL_SUCCESS(log, error)) return false;
  error = clCompileProgram(out_program, 1, &m_cl_device_id, nullptr, 0, nullptr,
                           nullptr, nullptr, nullptr);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  m_openPrograms.push_back(out_program);
  return true;
}

bool test_base_opencl::create_built_program(const std::string &source,
                                            cl_program &out_program,
                                            logger &log) {
  assert(!source.empty());

  const char *source_c = source.c_str();
  const size_t sourceSize = source.length();
  assert(source_c != nullptr);

  cl_int error = CL_SUCCESS;
  out_program = clCreateProgramWithSource(m_cl_context, 1, &source_c,
                                          &sourceSize, &error);
  if (!CHECK_CL_SUCCESS(log, error)) return false;
  error = clBuildProgram(out_program, 1, &m_cl_device_id, nullptr, nullptr,
                         nullptr);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  m_openPrograms.push_back(out_program);
  return true;
}

/**
 */
bool test_base_opencl::create_kernel(const cl_program &clProgram,
                                     const std::string &name,
                                     cl_kernel &out_kernel, logger &log) {
  assert(clProgram != nullptr);
  assert(!name.empty());

  cl_int error = CL_SUCCESS;
  out_kernel = clCreateKernel(clProgram, name.c_str(), &error);
  if (!CHECK_CL_SUCCESS(log, error)) return false;

  assert(out_kernel);

  m_openKernels.push_back(out_kernel);
  return true;
}

bool test_base_opencl::create_sampler(cl_sampler &outSampler, logger &log) {
  cl_int error = CL_SUCCESS;
  outSampler = clCreateSampler(m_cl_context, 0, CL_ADDRESS_REPEAT,
                               CL_FILTER_LINEAR, &error);
  return CHECK_CL_SUCCESS(log, error);
}

/** called after this test has executed
 */
void test_base_opencl::cleanup() {
  clReleaseCommandQueue(m_cl_command_queue);
  clReleaseContext(m_cl_context);

  size_t i = 0;
  for (i = 0; i < m_openKernels.size(); i++) clReleaseKernel(m_openKernels[i]);
  m_openKernels.clear();

  for (i = 0; i < m_openPrograms.size(); i++)
    clReleaseProgram(m_openPrograms[i]);
  m_openPrograms.clear();
}

}  // namespace util
}  // namespace sycl_cts
