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
  auto queue = get_cts_object::queue();

  INFO("Checking queue is using CUDA backend");
  REQUIRE(queue.get_backend() == sycl::backend::cuda);

  // Check for types, sycl::vec, and sycl::marray
  for_all_types<run_all_tests>(get_types(), queue);

  // Check for structs and classes
  run_all_tests<s1> tests_s1;
  tests_s1(queue, "struct");

  run_all_tests<c1> tests_c1;
  tests_c1(queue, "class with default constructor");

  run_all_tests<c2> tests_c2;
  tests_c2(queue, "class with default constructor");
#else
  SKIP("CUDA backend is not supported");
#endif  // SYCL_BACKEND_CUDA
}
