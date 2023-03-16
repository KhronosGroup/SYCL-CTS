/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
//  Provides tests for sycl::errc_for
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "exceptions.h"

namespace exceptions_errc_for_test {

bool check_opencl_supporting(const sycl::queue &q) {
  bool opencl_supported{false};
#ifdef SYCL_BACKEND_OPENCL
  opencl_supported = q.get_backend() == sycl::backend::opencl;
#endif  // SYCL_BACKEND_OPENCL
  return opencl_supported;
}

template <template <sycl::backend> class arg>
struct check_template_exists {};

// !FIXME Disabled for dpcpp until error_category_for() is implemented according
// to SYCL 2020 specification (4.13.2. Exception class interface)
// https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:exception.class
DISABLED_FOR_TEST_CASE(DPCPP)
("Check sycl::exception sycl::errc_for enum", "[exception]")({
  if (false ==
      check_opencl_supporting(sycl_cts::util::get_cts_object::queue())) {
    SKIP("OpenCL backend is not supported on this device");
    return;
  }
  // check that sycl::errc_for exist
  check_template_exists<sycl::errc_for>();

#ifdef SYCL_BACKEND_OPENCL
  using sycl_errc_enum_t = sycl::errc_for<sycl::backend::opencl>;
  // check that sycl::errc_for is enum and scoped enum
  {
    INFO("sycl::errc_for is not enum");
    CHECK(true == std::is_enum_v<sycl_errc_enum_t>);
  }
  {
    INFO(
        "sycl::errc_for is not a scoped enum cause it can be implicitly "
        "converted to int");
    CHECK(false ==
          std::is_convertible_v<sycl_errc_enum_t,
                                std::underlying_type<sycl_errc_enum_t>>);
  }
  const auto errc_value{static_cast<sycl_errc_enum_t>(0)};
  std::error_code err_code(errc_value,
                           sycl::error_category_for<sycl::backend::opencl>());
  {
    INFO(
        "error_code::default_error_condition() is not equal to "
        "std::error_condition");
    CHECK(err_code.default_error_condition() ==
          std::error_condition(
              errc_value, sycl::error_category_for<sycl::backend::opencl>()));
  }
  {
    INFO("sycl::errc_for is not a error code enumeration");
    CHECK(std::is_error_code_enum_v<sycl_errc_enum_t>);
  }
  {
    INFO("sycl::errc_for is a error condition enumeration");
    CHECK(false == std::is_error_condition_enum_v<sycl_errc_enum_t>);
  }
  {
    INFO(
        "sycl::error_category_for<sycl::backend::opencl> name is not equal to "
        "\"opencl\"");
    CHECK(sycl::error_category_for<sycl::backend::opencl>().name() == "opencl");
  }
#endif  // SYCL_BACKEND_OPENCL
});

}  // namespace exceptions_errc_for_test
