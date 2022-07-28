/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests multi_ptr comparison operators for double type
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_comparison_op.h"

namespace multi_ptr_comparison_op_fp64 {

TEST_CASE("multi_ptr comparison operators. fp64 type", "[multi_ptr]") {
  using namespace multi_ptr_comparison_op;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

  check_multi_ptr_comparison_op_for_type<double>{}("double");
}

}  // namespace multi_ptr_comparison_op_fp64
