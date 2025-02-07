/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests the API for sycl::handler::copy for double
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "handler_copy_common.h"

#include "catch2/catch_test_macros.hpp"

namespace handler_copy_fp64 {
using namespace handler_copy_common;

// Disabled: SimSYCL does not implement copies between accessors of different
// dimensionality
DISABLED_FOR_TEST_CASE(SimSYCL)
("Tests the API for sycl::handler::copy for double", "[handler]")({
  auto queue = util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

  log_helper lh;
  test_all_variants<double>(lh, queue);
  test_all_variants<sycl::double16>(lh, queue);
});

}  // namespace handler_copy_fp64
