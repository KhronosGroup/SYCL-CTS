/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr implicit conversions for sycl::half type
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_implicit_conversions.h"

namespace multi_ptr_implicit_conversions_fp16 {

TEST_CASE("multi_ptr implicit conversions. fp16 types", "[multi_ptr]") {
  using namespace multi_ptr_implicit_conversions;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_implicit_convert_for_type<sycl::half>("sycl::half");
}

}  // namespace multi_ptr_implicit_conversions_fp16
