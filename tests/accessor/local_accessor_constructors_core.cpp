/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::local_accessor constructors test for generic types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::local_accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "local_accessor_constructors.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace local_accessor_constructors_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::local_accessor constructors. core types", "[accessor]")({
  using namespace local_accessor_constructors;
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_all_types_vectors_marray<run_local_constructors_test>(types);
});
}  // namespace local_accessor_constructors_core
