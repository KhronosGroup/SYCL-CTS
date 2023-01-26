/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for interaction reductions with double variable type without
//  identity param.
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "reduction_without_identity_param_common.h"

namespace reduction_without_identity_param_fp64 {

using namespace sycl_cts;
using namespace reduction_without_identity;
using namespace reduction_common;

// FIXME: re-enable when compilation failure for double, nd_range and reduction
// is fixed.
DISABLED_FOR_TEST_CASE(DPCPP)
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
