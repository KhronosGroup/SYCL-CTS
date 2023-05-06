/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor api test for the double type
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::accessor is implemented
// Issue link https://github.com/intel/llvm/issues/8301
// Will be fixed with PR https://github.com/intel/llvm/pull/8874
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
#include "accessor_common.h"
#include "generic_accessor_api_common.h"
#endif

namespace generic_accessor_api_fp64 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor api. fp64 type", "[accessor]")({
  using namespace generic_accessor_api_common;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_generic_api_for_type, double>("double");
#else
  run_generic_api_for_type<double>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});
}  // namespace generic_accessor_api_fp64
