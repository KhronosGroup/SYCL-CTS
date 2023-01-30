/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for reductions with identity parameter for arithmetic
//  scalar types.
//
*******************************************************************************/

#include "reduction_with_identity_param.h"

namespace reduction_with_identity_param_core {
using namespace sycl_cts;

TEST_CASE("reduction_with_identity_param_core", "[reduction]") {
  auto queue = util::get_cts_object::queue();

  for_all_types<reduction_with_identity_param::run_test_for_type>(
      reduction_common::scalar_types, queue);
}
}  // namespace reduction_with_identity_param_core
