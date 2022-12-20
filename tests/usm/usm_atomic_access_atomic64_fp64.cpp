/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification to atomic access for USM allocations with
//  double-precision as their underlying type.
//
*******************************************************************************/

#include "usm_atomic_access.h"

#include "catch2/catch_test_macros.hpp"

namespace usm_atomic_access_atomic64_fp64 {
using namespace sycl_cts;

TEST_CASE("Tests for usm atomics with double-precision floating points",
          "[usm][atomic][atomic64][fp64]") {
  auto queue{util::get_cts_object::queue()};
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point "
        "operations. Skipping the test case.");
    return;
  }
  for_all_types<usm_atomic_access::run_all_tests>(
      usm_atomic_access::get_fp64_type(), queue,
      usm_atomic_access::with_atomic64);
}
}  // namespace usm_atomic_access_atomic64_fp64
