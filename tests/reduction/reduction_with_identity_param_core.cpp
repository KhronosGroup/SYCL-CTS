/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for arithmetic
//  scalar types.
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "reduction_with_identity_param.h"
#endif

namespace reduction_with_identity_param_core {
using namespace sycl_cts;

// FIXME: re-enable when span reduction is supported in ComputeCpp and
// sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(ComputeCpp, hipSYCL)
("reduction_with_identity_param_core", "[reduction]")({
  auto queue = util::get_cts_object::queue();

  for_all_types<reduction_with_identity_param::run_test_for_type>(
      reduction_common::scalar_types, queue);
});
}  // namespace reduction_with_identity_param_core
