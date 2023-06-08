/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for host_accessor properties with generic types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::host_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
#include "accessor_common.h"
#include "host_accessor_properties.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace host_accessor_properties_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::host_accessor properties. core types", "[accessor]")({
  using namespace host_accessor_properties;
  common_run_tests<run_host_properties_tests>();
});

}  // namespace host_accessor_properties_core
