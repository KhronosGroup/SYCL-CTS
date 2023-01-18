/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref constructors test for atomic64 types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "atomic_ref_constructors.h"

namespace atomic_ref_constructors_core_atomic64 {

TEST_CASE("sycl::atomic_ref constructors. atomic64 types", "[atomic_ref]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::atomic64)) {
    WARN(
        "Device does not support atomic64 operations. "
        "Skipping the test case.");
    return;
  }
  const auto type_pack = atomic_ref_tests_common::get_atomic64_types();
  for_all_types<atomic_ref_constructors::run_test>(type_pack);
}

}  // namespace atomic_ref_constructors_core_atomic64
