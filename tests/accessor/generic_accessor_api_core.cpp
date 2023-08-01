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
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "accessor_common.h"
#include "generic_accessor_api_common.h"

using namespace generic_accessor_api_common;
#endif

namespace generic_accessor_api_core {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Generic sycl::accessor api. core types", "[accessor]", test_combinations)({
  using namespace generic_accessor_api_common;
  common_run_tests<run_generic_api_for_type, TestType>();
});

}  // namespace generic_accessor_api_core
