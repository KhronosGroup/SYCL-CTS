/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for double.
//
*******************************************************************************/

#include "reduction_with_identity_param.h"

namespace reduction_with_identity_param_fp64 {
using namespace sycl_cts;

TEST_CASE("reduction_with_identity_param_fp64", "[reduction]") {
  auto queue = util::get_cts_object::queue();

  if (queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations");
  }
  reduction_with_identity::run_test_for_type<double>()(queue, "double");
}
} // reduction_with_identity_param_fp64