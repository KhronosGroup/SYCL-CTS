/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for sycl::half.
//
*******************************************************************************/

#include "reduction_with_identity_param.h"

namespace reduction_with_identity_param_fp16 {
using namespace sycl_cts;

TEST_CASE("reduction_with_identity_param_fp16", "[reduction]") {
  auto queue = util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    SKIP("Device does not support half precision floating point operations");
  }

  reduction_with_identity_param::run_test_for_type<sycl::half>()(queue,
                                                                 "sycl::half");
}
}  // namespace reduction_with_identity_param_fp16
