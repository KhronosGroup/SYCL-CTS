/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref constructors test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "atomic_ref_constructors.h"

namespace atomic_ref_constructors_core {

TEST_CASE("sycl::atomic_ref constructors. core types", "[atomic_ref]") {
  const auto types = atomic_ref_tests_common::get_conformance_type_pack();
  for_all_types<atomic_ref_constructors::run_test>(types);
}

}  // namespace atomic_ref_constructors_core
