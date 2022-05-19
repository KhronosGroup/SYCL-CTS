/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_TEST_BASE_CUDA_H
#define __SYCLCTS_UTIL_TEST_BASE_CUDA_H

#include "test_base.h"

#include <cuda.h>
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#include <sycl/sycl.hpp>

// conformance test suite namespace
namespace sycl_cts {
namespace util {

/** Common base class for CUDA inter operation tests
 */
class test_base_cuda : public sycl_cts::util::test_base {
 public:
  /** ctor
   */
  test_base_cuda();

  /** virtual destructor
   */
  virtual ~test_base_cuda() {}

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

  /* instances of OpenCL objects */
  CUdevice m_cu_device;
  std::vector<CUdevice> m_cu_platform;
  CUstream m_cu_stream;
  CUevent m_cu_event;

  /*  */

};  // class test_base

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_TEST_BASE_CUDA_H
