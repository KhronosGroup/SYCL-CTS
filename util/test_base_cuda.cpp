/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "test_base_cuda.h"
#include "../tests/common/cts_selector.h"
#include "../tests/common/get_cts_object.h"
#include "../tests/common/macros.h"

#ifdef _MSC_VER
#include <windows.h>
#else
#include <unistd.h>
#endif
#ifdef SYCL_BACKEND_CUDA
// conformance test suite namespace
namespace sycl_cts {
namespace util {

/** constructor which explicitly sets the OpenCL objects
 *  to nullptrs
 */
test_base_cuda::test_base_cuda() {}

/** called before this test is executed
 *  @param log for emitting test notes and results
 */
bool test_base_cuda::setup(logger &log) {
  /* get the OpenCLHelper object */
  auto queue = util::get_cts_object::queue();
  if (queue.get_backend() != sycl::backend::cuda) {
    WARN(
        "CUDA interoperability part is not supported on non-CUDA backend "
        "types");
    return false;
  }

  cts_selector ctsSelector;
  const auto ctsContext = util::get_cts_object::context(ctsSelector);

  if (ctsContext.get_devices().empty()) {
    FAIL(log, "Unable to retrieve list of devices via cts_selector");
    return false;
  }
  const auto ctsDevice = ctsContext.get_devices()[0];

  return true;
}

void test_base_cuda::cleanup() {}

}  // namespace util
}  // namespace sycl_cts
#endif
