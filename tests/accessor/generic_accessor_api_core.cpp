/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor api test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "accessor_common.h"
#include "generic_accessor_api_common.h"
#endif

namespace generic_accessor_api_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("Generic sycl::accessor api. core types", "[accessor]")({
  using namespace generic_accessor_api_common;
  common_run_tests<run_generic_api_for_type>();
});

}  // namespace generic_accessor_api_core
