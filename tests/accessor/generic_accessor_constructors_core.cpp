/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor constructors test for generic types
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "accessor_common.h"
#include "catch2/catch_test_macros.hpp"
#include "generic_accessor_constructors.hpp"

namespace generic_accessor_constructors_core {
using namespace generic_accessor_constructors;

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor constructors. core types", "[accessor]")({
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_all_types_vectors_marray<run_generic_constructors_test>(types);
});

}  // namespace generic_accessor_constructors_core
