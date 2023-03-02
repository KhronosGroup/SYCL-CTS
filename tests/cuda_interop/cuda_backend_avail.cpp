/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Provide verification for CUDA backend availability
//
*******************************************************************************/

#include "cuda_helper.hpp"

int cuda_selector(const sycl::device& dev) const override {
  return (dev.get_platform().get_info<sycl::info::platform::name>().find(
              "CUDA") != std::string::npos)
             ? 1
             : -1;
}

TEST_CASE("CUDA backend availability test") {
#ifdef SYCL_BACKEND_CUDA
  sycl::queue q{cuda_selector};

  INFO("Checking sycl::backend::cuda is available");
  REQUIRE(q.get_backend() == sycl::backend::cuda);
#else
  SKIP("CUDA backend is not supported");
#endif  // SYCL_BACKEND_CUDA
}
