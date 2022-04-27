/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor constructors test for generic types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "generic_accessor_constructors.hpp"
#endif

#include "../common/disabled_for_test_case.h"

namespace generic_accessor_constructors_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor constructors. core types", "[accessor]")({
  using namespace generic_accessor_constructors;
  
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_all_types_vectors_marray<run_generic_constructors_test>(types);
});

}  // namespace generic_accessor_constructors_core
