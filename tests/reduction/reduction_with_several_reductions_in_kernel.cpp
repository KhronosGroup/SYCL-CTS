/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for several reductions in one kernel.
//
*******************************************************************************/

#include "reduction_with_several_reductions_in_kernel.h"

namespace reduction_with_several_reductions_in_kernel {
using namespace sycl_cts;

TEST_CASE("reduction_with_several_reductions_in_kernel", "[reduction]") {
  auto queue = util::get_cts_object::queue();
  reduction_with_several_reductions_in_kernel_h::run_all_tests(queue);
}
}  // namespace reduction_with_several_reductions_in_kernel
