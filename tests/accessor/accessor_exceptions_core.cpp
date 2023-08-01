/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor exceptions.
//
//  This test provides verifications that exception really has been thrown for
//  generic accessor, host_accessor and local_accessor with generic types.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "accessor_exceptions.h"

using namespace accessor_exceptions_test;
using namespace accessor_tests_common;
#endif

namespace accessor_exceptions_test_core {
using namespace sycl_cts;

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Generic sycl::accessor constructor exceptions test. Core types.",
 "[accessor]", test_combinations)({
  common_run_tests<run_tests_with_types, generic_accessor, TestType>();
});

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("sycl::local_accessor constructor exceptions test. Core types.", "[accessor]",
 test_combinations)({
  common_run_tests<run_tests_with_types, local_accessor, TestType>();
});

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("sycl::host_accessor constructor exceptions test. Core types.", "[accessor]",
 test_combinations)({
  common_run_tests<run_tests_with_types, host_accessor, TestType>();
});

}  // namespace accessor_exceptions_test_core
