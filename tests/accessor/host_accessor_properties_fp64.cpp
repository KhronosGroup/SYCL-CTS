/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for host_accessor properties with double type
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::host_accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "host_accessor_properties.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace host_accessor_properties_fp64 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::host_accessor properties. fp64 type", "[accessor]")({
  using namespace host_accessor_properties;
  auto queue = sycl_cts::util::get_cts_object::queue();
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_host_properties_tests, double>("double");
#else
  run_host_properties_tests<double>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});
}  // namespace host_accessor_properties_fp64
