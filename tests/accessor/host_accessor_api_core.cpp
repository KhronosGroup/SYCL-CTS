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
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "host_accessor_api_common.h"
#endif

namespace host_accessor_api_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::host_accessor api. core types", "[accessor]")({
  using namespace host_accessor_api_common;
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_host_accessor_api_for_type>(types);
  for_all_dev_copyable_containers<run_host_accessor_api_for_type>(types);
});

}  // namespace host_accessor_api_core
