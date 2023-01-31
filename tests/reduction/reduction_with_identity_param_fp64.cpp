/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for double.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "reduction_with_identity_param.h"
#endif

namespace reduction_with_identity_param_fp64 {

// FIXME: re-enable when span reduction is supported in ComputeCpp and
// sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(ComputeCpp, hipSYCL)
("reduction_with_identity_param_fp64", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations");
  }
  reduction_with_identity_param::run_test_for_type<double>()(queue, "double");
});
}  // namespace reduction_with_identity_param_fp64
