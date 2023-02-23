/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor api test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::accessor is implemented
// Issue links https://github.com/intel/llvm/issues/8298
// https://github.com/intel/llvm/issues/8299
// https://github.com/intel/llvm/issues/8301
// https://github.com/intel/llvm/issues/8302
// PR link https://github.com/intel/llvm/pull/8069
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
#include "accessor_common.h"
#include "generic_accessor_api_common.h"
#endif

namespace generic_accessor_api_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor api. core types", "[accessor]")({
  using namespace generic_accessor_api_common;
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_generic_api_for_type>(types);
  for_all_device_copyable_std_containers<run_generic_api_for_type>(types);
});

}  // namespace generic_accessor_api_core
