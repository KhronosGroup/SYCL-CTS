/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2025 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include <sycl/sycl.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../../common/get_cts_object.h"

#include <type_traits>
#include <vector>

namespace num_compute_units::tests {

TEST_CASE("Test for info::device::num_compute_units descriptor",
          "[oneapi_num_compute_units]") {
#ifndef SYCL_EXT_ONEAPI_NUM_COMPUTE_UNITS
  SKIP(
      "The sycl_ext_oneapi_num_compute_units device extension is not supported "
      "by an implementation");
#else

  sycl::device dev = sycl_cts::util::get_cts_object::device();

  STATIC_REQUIRE(
      std::is_same_v<
          size_t,
          decltype(dev.get_info<
                   sycl::ext::oneapi::info::device::num_compute_units>())>);

  INFO("Checking that query returns a result greater than or equal to 1");
  REQUIRE(dev.get_info<sycl::ext::oneapi::info::device::num_compute_units>() >=
          1);

#endif
}

}  // namespace num_compute_units::tests
