/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr local_accessor constructor for sycl::half type
//
*******************************************************************************/

#include "multi_ptr_local_accessor_constructor.h"

namespace multi_ptr_accessor_local_constructor_fp16 {

TEST_CASE("constructor multi_ptr(local_accessor<T, dims>). fp16 type",
          "[multi_ptr]") {
  using namespace multi_ptr_local_accessor_constructors;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_local_accessor_cnstr_for_type<sycl::half>{}("sycl::half");
}

}  // namespace multi_ptr_accessor_local_constructor_fp16
