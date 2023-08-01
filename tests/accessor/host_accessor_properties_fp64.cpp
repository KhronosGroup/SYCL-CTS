/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for host_accessor properties with double type
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::host_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "accessor_common.h"
#include "host_accessor_properties.h"

using namespace host_accessor_properties;
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace host_accessor_properties_fp64 {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("sycl::host_accessor properties. fp64 type", "[accessor]", test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_host_properties_tests, double, TestType>(
      "double");
#else
  run_host_properties_tests<double, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});
}  // namespace host_accessor_properties_fp64
