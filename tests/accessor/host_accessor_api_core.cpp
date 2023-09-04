/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::host_accessor api test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::host_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "accessor_common.h"
#include "host_accessor_api_common.h"

using namespace host_accessor_api_common;
#endif

namespace host_accessor_api_core {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("sycl::host_accessor api. core types", "[accessor]", test_combinations)({
  common_run_tests<run_host_accessor_api_for_type, TestType>();
});

}  // namespace host_accessor_api_core
