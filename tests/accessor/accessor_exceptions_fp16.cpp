/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor exceptions.
//
//  This test provides verifications that exception really has been thrown for
//  generic accessor, host_accessor and local_accessor with sycl::half type.
//
*******************************************************************************/
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)

#include "accessor_exceptions.hpp"

using namespace accessor_exceptions_test;
using namespace accessor_tests_common;
#endif

namespace accessor_exceptions_test_fp16 {
using namespace sycl_cts;

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor constructor exceptions test.", "[accessor]")({
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  run_tests_with_types<sycl::half, generic_accessor>{}("sycl::half");
#else
  for_type_vectors_marray<run_tests_with_types, sycl::half, generic_accessor>(
      "sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::local_accessor  constructor exceptions test.", "[accessor]")({
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  run_tests_with_types<sycl::half, local_accessor>{}("sycl::half");
#else
  for_type_vectors_marray<run_tests_with_types, sycl::half, local_accessor>(
      "sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::host_accessor constructor exceptions test.", "[accessor]")({
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  run_tests_with_types<sycl::half, host_accessor>{}("sycl::half");
#else
  for_type_vectors_marray<run_tests_with_types, sycl::half, host_accessor>(
      "sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});
}  // namespace accessor_exceptions_test_fp16
