/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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
