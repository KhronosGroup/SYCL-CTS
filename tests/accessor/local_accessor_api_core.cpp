/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::local_accessor api test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::local_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
#include "accessor_common.h"
#include "local_accessor_api_common.h"

using namespace local_accessor_api_common;
#endif

namespace local_accessor_api_core {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::local_accessor api. core types", "[accessor]",
 test_combinations)({ common_run_tests<run_local_api_for_type, TestType>(); });

}  // namespace local_accessor_api_core
