/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr explicit conversions for sycl::half type
//
*******************************************************************************/

#include "multi_ptr_explicit_conversions.h"

namespace multi_ptr_explicit_conversions_fp16 {

TEST_CASE("multi_ptr explicit conversions. fp16 type", "[multi_ptr]") {
  using namespace multi_ptr_explicit_conversions;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_explicit_convert_for_type<sycl::half>{}("sycl::half");
}

}  // namespace multi_ptr_explicit_conversions_fp16
