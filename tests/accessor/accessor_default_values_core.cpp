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
#include "accessor_common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "accessor_default_values.h"

using namespace accessor_default_values_test;
using namespace accessor_tests_common;
#endif

namespace accessor_default_values_test_core {
using namespace sycl_cts;

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Accessors constructor default values test core types.", "[accessor]",
 test_combinations)({ common_run_tests<run_tests, TestType>(); });

}  // namespace accessor_default_values_test_core
