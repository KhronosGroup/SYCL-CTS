/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::host_accessor api test for the double type
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::host_accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "host_accessor_api_common.h"
#endif

namespace host_accessor_api_fp64 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::host_accessor api. fp64 type", "[accessor]")({
  using namespace host_accessor_api_common;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  run_host_accessor_api_for_type<double>{}("double");
#else
  for_type_vectors_marray<run_host_accessor_api_for_type, double>("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});
}  // namespace host_accessor_api_fp64
