/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor api test for the double type
//
*******************************************************************************/
#include "accessor_common.h"
#include "catch2/catch_test_macros.hpp"
#include "generic_accessor_api_common.h"

namespace generic_accessor_api_fp64 {
using namespace generic_accessor_api_common;

TEST_CASE("Generic sycl::accessor api. fp64 type", "[accessor]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    run_generic_api_for_type<double>{}("double");
#else
    for_type_vectors_marray<run_generic_api_for_type, double>("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  }
}
}  // namespace generic_accessor_api_fp64
