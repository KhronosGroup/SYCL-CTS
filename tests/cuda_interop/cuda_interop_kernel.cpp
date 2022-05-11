/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:  (c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_BACKEND_CUDA
#include "../../util/test_base_cuda.h"
#include "cuda_interop_kernel_func_tests.hpp"
#endif

#define TEST_NAME cuda_interop_kernel

namespace cuda_interop_kernel__ {

using namespace sycl_cts;

/** tests the get_native() methods for CUDA inter-op
 */
class TEST_NAME :
#ifdef SYCL_BACKEND_CUDA
    public sycl_cts::util::test_base_cuda
#else
    public util::test_base
#endif
{
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
#ifdef SYCL_BACKEND_CUDA
    {
      auto queue = util::get_cts_object::queue();
      if (queue.get_backend() != sycl::backend::cuda) {
        WARN(
            "CUDA interoperability part is not supported on non-CUDA "
            "backend types");
        return;
      }
      cts_selector ctsSelector;

      // Check for types, sycl::vec, and sycl::marray
      for_all_types<run_all_tests>(get_types(), queue, log);

      // Check for structs and classes
      run_all_tests<s1> tests_s1;
      tests_s1(queue, log, "struct");

      run_all_tests<c1> tests_c1;
      tests_c1(queue, log, "class with default constructor");

      run_all_tests<c2> tests_c2;
      tests_c2(queue, log, "class with default constructor");
    }
#else
    log.note("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_BACKEND_CUDA
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace cuda_interop_kernel__
