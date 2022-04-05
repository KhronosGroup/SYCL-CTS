/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor test for the double type
//
*******************************************************************************/
#include "accessor_common.h"
#include "catch2/catch_test_macros.hpp"
#include "generic_accessor_constructors.hpp"

namespace generic_accessor_constructors_fp64 {
using namespace generic_accessor_constructors;

TEST_CASE("Generic sycl::accessor constructors. fp64 type", "[accessor]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
    const auto types = get_fp64_type();
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    run_generic_constructors_test<double>{}("double");
#else
    for_type_vectors_marray<run_generic_constructors_test, double>("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  }
}
}  // namespace generic_accessor_constructors_fp64
