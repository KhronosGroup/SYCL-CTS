/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::host_accessor linearization test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::host_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "accessor_common.h"
#include "host_accessor_linearization.h"
#endif

namespace host_accessor_liniarization_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::host_accessor linearization. core types", "[accessor]")({
  using namespace host_accessor_linearization;
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_host_linearization_for_type>(types);
});

}  // namespace host_accessor_liniarization_core
