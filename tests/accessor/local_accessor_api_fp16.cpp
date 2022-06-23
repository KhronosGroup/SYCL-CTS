/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::local_accessor api test for the sycl::half type
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::local_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
#include "accessor_common.h"
#include "local_accessor_api_common.h"
#endif

namespace local_accessor_api_fp16 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::local_accessor api. fp16 type", "[accessor]")({
  using namespace local_accessor_api_common;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
  run_local_api_for_type<sycl::half>{}("sycl::half");
#else
  for_type_vectors_marray<run_local_api_for_type, sycl::half>("sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});
}  // namespace local_accessor_api_fp16
