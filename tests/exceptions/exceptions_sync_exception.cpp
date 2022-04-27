/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for for synchronous exceptions mechanism
//
*******************************************************************************/

#include <catch2/catch_template_test_macros.hpp>

#include "../common/common.h"

using namespace sycl_cts;

TEST_CASE("Check an empty queue::submit() doesn't result in exceptions.",
          "[exception]") {
  auto q = util::get_cts_object::queue();
  CHECK_NOTHROW(q.submit([&](sycl::handler &cgh) {}).wait());
}
