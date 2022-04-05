/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor api test for generic types
//
*******************************************************************************/
#include "accessor_common.h"
#include "catch2/catch_test_macros.hpp"
#include "generic_accessor_api_common.h"

namespace generic_accessor_api_core {
using namespace generic_accessor_api_common;

TEST_CASE("Generic sycl::accessor api. core types", "[accessor]") {
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_all_types_vectors_marray<run_generic_api_for_type>(types);
}

}  // namespace generic_accessor_api_core
