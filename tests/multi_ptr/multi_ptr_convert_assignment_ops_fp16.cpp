/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests multi_ptr convert assignment operators for sycl::half type
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_convert_assignment_ops.h"

namespace multi_ptr_convert_assignment_ops_fp16 {

TEST_CASE("Convert assignment operators. fp16 type", "[multi_ptr]") {
  using namespace multi_ptr_convert_assignment_ops;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_convert_assign_for_type<sycl::half>{}("sycl::half");
}

}  // namespace multi_ptr_convert_assignment_ops_fp16
