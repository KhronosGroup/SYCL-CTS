/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::local_accessor api test for the double type
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::local_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "accessor_common.h"
#include "local_accessor_api_common.h"

using namespace local_accessor_api_common;
#endif

namespace local_accessor_api_fp64 {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("sycl::local_accessor api. fp64 type", "[accessor]", test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_local_api_for_type, double, TestType>("double");
#else
  run_local_api_for_type<double, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});
}  // namespace local_accessor_api_fp64
