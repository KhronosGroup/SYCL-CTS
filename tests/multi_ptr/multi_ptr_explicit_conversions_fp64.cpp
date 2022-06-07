/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr explicit conversions for dpuble type
//
*******************************************************************************/

#include "multi_ptr_explicit_conversions.h"

namespace multi_ptr_explicit_conversions_fp64 {

TEST_CASE("multi_ptr explicit conversions. fp64 type", "[multi_ptr]") {
  using namespace multi_ptr_explicit_conversions;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_explicit_convert_for_type<double>{}("double");
}

}  // namespace multi_ptr_explicit_conversions_fp64
