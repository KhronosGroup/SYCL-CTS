/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#include "cuda_helper.hpp"
#include "cuda_interop_kernel_func_tests.hpp"

using namespace sycl_cts::util;

TEST_CASE("CUDA interop kernel test") {
#ifdef SYCL_BACKEND_CUDA
  {
    auto queue = get_cts_object::queue();
    if (queue.get_backend() != sycl::backend::cuda) {
      WARN(
          "CUDA interoperability part is not supported on non-CUDA "
          "backend types");
      return;
    }

    // Check for types, sycl::vec, and sycl::marray
    for_all_types<run_all_tests>(get_types(), queue);

    // Check for structs and classes
    run_all_tests<s1> tests_s1;
    tests_s1(queue, "struct");

    run_all_tests<c1> tests_c1;
    tests_c1(queue, "class with default constructor");

    run_all_tests<c2> tests_c2;
    tests_c2(queue, "class with default constructor");
  }
#else
  INFO("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_BACKEND_CUDA
}
