/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref constructors test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable for computecpp when
// sycl::access::address_space::generic_space and possibility of a SYCL kernel
// with an unnamed type are implemented in computecpp, re-enable for hipsycl
// when sycl::info::device::atomic_memory_order_capabilities and
// sycl::info::device::atomic_memory_scope_capabilities are implemented in
// hipsycl
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "atomic_ref_constructors.h"

#endif

namespace atomic_ref_constructors_core {

DISABLED_FOR_TEST_CASE(ComputeCpp, hipSYCL)
("sycl::atomic_ref constructors. core types", "[atomic_ref]")({
  const auto types = atomic_ref_tests_common::get_conformance_type_pack();
  for_all_types<atomic_ref_constructors::run_test>(types);
});

}  // namespace atomic_ref_constructors_core
