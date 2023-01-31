/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for interaction reductions with double variable type without
//  identity param.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "reduction_without_identity_param_common.h"
#endif

namespace reduction_without_identity_param_fp64 {

using namespace sycl_cts;
using namespace reduction_without_identity_param_common;
using namespace reduction_common;

// FIXME: re-enable when compilation failure for reduction with custom type is
// fixed and span reduction is supported in ComputeCpp and sycl::reduction is
// implemented in hipSYCL
DISABLED_FOR_TEST_CASE(DPCPP, ComputeCpp, hipSYCL)
("reduction_without_identity_param", "[reduction]")({
  auto queue = util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations");
  }

  run_tests_for_all_functors<double, run_test_without_property>()(range, queue,
                                                                  "double");
  run_tests_for_all_functors<double, run_test_with_property>()(nd_range, queue,
                                                               "double");
});

}  // namespace reduction_without_identity_param_fp64
