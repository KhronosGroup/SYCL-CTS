/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr local_accessor constructor for type double
//
*******************************************************************************/

#include "multi_ptr_local_accessor_constructor.h"

namespace multi_ptr_accessor_local_constructor_fp64 {

TEST_CASE("constructor multi_ptr(local_accessor<T, dims>). fp64 type",
          "[multi_ptr]") {
  using namespace multi_ptr_local_accessor_constructors;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_local_accessor_cnstr_for_type<double>{}("double");
}

}  // namespace multi_ptr_accessor_local_constructor_fp64
