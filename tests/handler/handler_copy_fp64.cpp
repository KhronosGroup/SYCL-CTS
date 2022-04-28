/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests the API for sycl::handler::copy for double
//
*******************************************************************************/

#include "handler_copy_common.h"

#include "catch2/catch_test_macros.hpp"

namespace handler_copy_fp64 {
using namespace handler_copy_common;

TEST_CASE("Tests the API for sycl::handler::copy for double", "[handler]") {
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
}

}  // namespace handler_copy_fp64
