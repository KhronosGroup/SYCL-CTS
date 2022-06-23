/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr implicit conversions for double type
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_implicit_conversions.h"

namespace multi_ptr_implicit_conversions_fp64 {

TEST_CASE("multi_ptr implicit conversions. fp64 type", "[multi_ptr]") {
  using namespace multi_ptr_implicit_conversions;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_implicit_convert_for_type<double>("double");
}

}  // namespace multi_ptr_implicit_conversions_core
