/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification to atomic access for USM allocations that underlying
//  type size lower than 64 byte.
//
*******************************************************************************/

#include "usm_atomic_access.h"

#include "catch2/catch_test_macros.hpp"

namespace usm_atomic_access_core {
using namespace sycl_cts;

TEST_CASE("Tests for usm atomics with atomic64 support", "[usm][atomic]") {
  auto queue{util::get_cts_object::queue()};

  for_all_types<usm_atomic_access::run_all_tests>(
      usm_atomic_access::get_nondouble_scalar_types(), queue,
      usm_atomic_access::without_atomic64);
}
}  // namespace usm_atomic_access_core
