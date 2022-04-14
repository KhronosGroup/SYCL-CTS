/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel from functor for type double
//
*******************************************************************************/

#include "invoke_template_kernels_common.h"

#include "catch2/catch_test_macros.hpp"

namespace invoke_template_kernels {
using namespace invoke_template_kernels_common;

TEST_CASE("Tests for sycl::kernel from functor for double", "[invoke]") {
  auto sycl_queue = util::get_cts_object::queue();
  if (!sycl_queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

  static const float test_double_value = 10;
  REQUIRE(test_kernel_functor(test_double_value, sycl_queue));
  sycl_queue.wait_and_throw();
}

}  // namespace invoke_template_kernels
