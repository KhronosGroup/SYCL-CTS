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
#endif

namespace local_accessor_api_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::local_accessor api. core types", "[accessor]")({
  using namespace local_accessor_api_common;
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_local_api_for_type>(types);
});

}  // namespace local_accessor_api_core
