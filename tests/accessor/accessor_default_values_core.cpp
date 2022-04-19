/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor default values.
//
//  This test provides verifications that template parameters has default values
//  for generic accessor, host_accessor and local_accessor with generic types.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)

#include "accessor_default_values.h"

using namespace accessor_default_values_test;
using namespace accessor_tests_common;
#endif

namespace accessor_default_values_test_core {
using namespace sycl_cts;

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Accessors constructor default values test core types.", "[accessor]")({
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_all_types_vectors_marray<run_tests, generic_accessor>(types);
});

}  // namespace accessor_default_values_test_core
