/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for generic accessor properties with generic types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "accessor_common.h"
#include "generic_accessor_properties.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace generic_accessor_properties_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("Generic sycl::accessor properties test. core types", "[accessor]")({
  using namespace generic_accessor_properties;
  common_run_tests<run_generic_properties_tests>();
});

}  // namespace generic_accessor_properties_core
