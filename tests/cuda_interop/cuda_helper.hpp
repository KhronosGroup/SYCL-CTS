/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#include "../common/common.h"
#include <stdexcept>

#ifdef SYCL_BACKEND_CUDA
#include <cuda.h>
#endif
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#include <sycl/sycl.hpp>

inline void cuda_check(CUresult error_code) {
  if (error_code != CUDA_SUCCESS) {
    const char* error_name;
    cuGetErrorName(error_code, &error_name);
    throw std::runtime_error(std::string("CUDA error:") + error_name);
  }
}

#define CUDA_CHECK(result) cuda_check(result)
