/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel from functor
//
*******************************************************************************/

#include "invoke_template_kernels_common.h"

#include "catch2/catch_test_macros.hpp"

namespace invoke_template_kernels {
using namespace invoke_template_kernels_common;

TEST_CASE("Tests for sycl::kernel from functor", "[invoke]") {
  auto sycl_queue = util::get_cts_object::queue();

  static const float test_float_value = 10;
  {
    INFO("float type");
    REQUIRE(test_kernel_functor(test_float_value, sycl_queue));
  }
  {
    INFO("int8_t type");
    REQUIRE(test_kernel_functor(static_cast<int8_t>(INT8_MAX), sycl_queue));
  }
  {
    INFO("int16_t type");
    REQUIRE(test_kernel_functor(static_cast<int16_t>(INT16_MAX), sycl_queue));
  }
  {
    INFO("int32_t type");
    REQUIRE(test_kernel_functor(static_cast<int32_t>(INT32_MAX), sycl_queue));
  }
  {
    INFO("int64_t type");
    REQUIRE(test_kernel_functor(static_cast<int64_t>(INT64_MAX), sycl_queue));
  }
  {
    INFO("uint8_t type");
    REQUIRE(test_kernel_functor(static_cast<uint8_t>(UINT8_MAX), sycl_queue));
  }
  {
    INFO("uint16_t type");
    REQUIRE(test_kernel_functor(static_cast<uint16_t>(UINT16_MAX), sycl_queue));
  }
  {
    INFO("uint32_t type");
    REQUIRE(test_kernel_functor(static_cast<uint32_t>(UINT32_MAX), sycl_queue));
  }
  {
    INFO("uint64_t type");
    REQUIRE(test_kernel_functor(static_cast<uint64_t>(UINT64_MAX), sycl_queue));
  }

  sycl_queue.wait_and_throw();
}

}  // namespace invoke_template_kernels
