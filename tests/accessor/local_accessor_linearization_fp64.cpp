/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for local_accessor linearization with double type
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::local_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "accessor_common.h"
#include "local_accessor_linearization.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace local_accessor_linearization_fp64 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::local_accessor linearization test. fp64 type", "[accessor]")({
  using namespace local_accessor_linearization;
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_local_linearization_for_type, double>("double");
#else
  run_local_linearization_for_type<double>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

}  // namespace local_accessor_linearization_fp64
